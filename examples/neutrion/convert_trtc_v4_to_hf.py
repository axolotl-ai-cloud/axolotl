#!/usr/bin/env python3
"""Expand a Fermion Research TRTC v4 arch-3 container into a plain HF Qwen3
checkpoint (bf16 safetensors) so it can be fine-tuned.

The container stores every transformer linear five-valued: three bit planes
(bp/bn/br) plus two per-row magnitudes, with w[r,c] = (bp-bn) * (br ? s_hi : s_lo);
embeddings/lm_head are per-row-scaled int8. Both are exactly representable as
dense matrices, and arch-3 is the Qwen3 topology (biasless QKV, per-head Q/K
RMSNorm before rope, SwiGLU), so the expansion is a 1:1 tensor rename.

Reader layout follows the container spec shipped in the model repo at
mlx/fermion_mlx/container.py.

    python neutrino_trtc_to_hf.py <container.bin> <out_dir> [--tokenizer-repo REPO]
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import Qwen3Config

MAGIC = 0x43545254
HDR_FMT = "<Iii9i2fi"
HDR_BYTES = struct.calcsize(HDR_FMT)
PROJS = ("q", "k", "v", "o", "gate", "up", "down")
HF_PROJ = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]
SHARD_BYTES = 4 * 1024**3


def _f32(mm, count, off):
    return np.frombuffer(mm, np.float32, count, off), off + 4 * count


def _trit_record(mm, off):
    rows, cols, cols_pad, row_stride = (
        int(v) for v in np.frombuffer(mm, np.int32, 4, off)
    )
    off += 16
    n = rows * row_stride
    planes = []
    for _ in range(3):  # bp, bn, br
        planes.append(np.frombuffer(mm, np.uint8, n, off).reshape(rows, row_stride))
        off += n
    slo, off = _f32(mm, rows, off)
    shi, off = _f32(mm, rows, off)
    off += 8 * rows  # rs_lo / rs_hi integrity sidecars
    has_bias = int(np.frombuffer(mm, np.int32, 1, off)[0])
    off += 4
    assert not has_bias, "arch-3 records must be bias-less"
    return (rows, cols, planes, slo, shi), off


def _int8_record(mm, off):
    rows, cols, cols_pad, row_stride = (
        int(v) for v in np.frombuffer(mm, np.int32, 4, off)
    )
    assert row_stride == cols_pad
    off += 16
    n = rows * row_stride
    weights = np.frombuffer(mm, np.int8, n, off).reshape(rows, row_stride)
    off += n
    scale, off = _f32(mm, rows, off)
    off += 4 * rows  # wsum
    return (rows, cols, weights, scale), off


def _dense_trit(rec):
    rows, cols, (bp, bn, br), slo, shi = rec
    pos = np.unpackbits(bp, axis=1, bitorder="little")[:, :cols].astype(np.int8)
    neg = np.unpackbits(bn, axis=1, bitorder="little")[:, :cols].astype(np.int8)
    hi = np.unpackbits(br, axis=1, bitorder="little")[:, :cols].astype(bool)
    mag = np.where(hi, shi[:, None], slo[:, None]).astype(np.float32)
    return torch.from_numpy((pos - neg).astype(np.float32) * mag).to(torch.bfloat16)


def _dense_int8(rec):
    rows, cols, weights, scale = rec
    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    step = 8192
    for i in range(0, rows, step):
        chunk = weights[i : i + step, :cols].astype(np.float32)
        chunk *= scale[i : i + step, None]
        out[i : i + step] = torch.from_numpy(chunk).to(torch.bfloat16)
    return out


def convert(container_path: Path, out_dir: Path, tokenizer_repo: str):
    mm = np.memmap(container_path, dtype=np.uint8, mode="r")
    (
        magic,
        version,
        arch,
        n_layers,
        hidden,
        n_heads,
        n_kv,
        head_dim,
        rot,
        inter,
        vocab,
        maxpos,
        eps,
        theta,
        eok,
    ) = struct.unpack(HDR_FMT, bytes(mm[:HDR_BYTES]))
    assert magic == MAGIC, f"bad magic {magic:#x}"
    assert (version, arch) == (4, 3), f"need TRTC v4 arch-3, got v{version} arch{arch}"
    assert rot == head_dim, "expansion assumes full-head_dim rope"
    print(
        f"TRTC v{version} arch{arch}: {n_layers}L hidden={hidden} "
        f"heads={n_heads}/{n_kv} head_dim={head_dim} inter={inter} "
        f"vocab={vocab} maxpos={maxpos} eps={eps} theta={theta} eok={eok}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    shards, index, shard_bytes, shard_tensors = [], {}, 0, {}

    def emit(name, tensor):
        nonlocal shard_bytes, shard_tensors
        if shard_bytes and shard_bytes + tensor.nbytes > SHARD_BYTES:
            shards.append(shard_tensors)
            shard_tensors, shard_bytes = {}, 0
        shard_tensors[name] = tensor.contiguous()
        shard_bytes += tensor.nbytes
        index[name] = len(shards)

    off = HDR_BYTES
    embed_in = None
    if eok == 2:
        embed_in, off = _int8_record(mm, off)
        assert (embed_in[0], embed_in[1]) == (vocab, hidden)
        emit("model.embed_tokens.weight", _dense_int8(embed_in))

    q_dim, kv_dim = n_heads * head_dim, n_kv * head_dim
    shapes = {
        "q": (q_dim, hidden),
        "k": (kv_dim, hidden),
        "v": (kv_dim, hidden),
        "o": (hidden, q_dim),
        "gate": (inter, hidden),
        "up": (inter, hidden),
        "down": (hidden, inter),
    }
    for li in range(n_layers):
        prefix = f"model.layers.{li}."
        for norm_name, count in (
            ("input_layernorm", hidden),
            ("post_attention_layernorm", hidden),
            ("self_attn.q_norm", head_dim),
            ("self_attn.k_norm", head_dim),
        ):
            vec, off = _f32(mm, count, off)
            emit(
                prefix + norm_name + ".weight",
                torch.from_numpy(vec.copy()).to(torch.bfloat16),
            )
        for proj in PROJS:
            rec, off = _trit_record(mm, off)
            assert (rec[0], rec[1]) == shapes[proj], f"L{li}.{proj}: {rec[:2]}"
            emit(prefix + HF_PROJ[proj] + ".weight", _dense_trit(rec))
        print(f"  layer {li + 1}/{n_layers}", end="\r", flush=True)
    print()

    final_norm, off = _f32(mm, hidden, off)
    emit("model.norm.weight", torch.from_numpy(final_norm.copy()).to(torch.bfloat16))
    embed_out, off = _int8_record(mm, off)
    assert (embed_out[0], embed_out[1]) == (vocab, hidden)
    emit("lm_head.weight", _dense_int8(embed_out))
    if embed_in is None:  # tied: the lm_head record serves both
        emit("model.embed_tokens.weight", _dense_int8(embed_out))
    assert off == mm.size, f"reader ended at {off}, file is {mm.size}"
    shards.append(shard_tensors)

    names = [
        f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        for i in range(len(shards))
    ]
    total = 0
    for name, tensors in zip(names, shards):
        total += sum(t.nbytes for t in tensors.values())
        save_file(tensors, str(out_dir / name), metadata={"format": "pt"})
        print(f"wrote {name} ({len(tensors)} tensors)")
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": {k: names[v] for k, v in index.items()},
            },
            indent=2,
        )
    )

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        max_position_embeddings=maxpos,
        rms_norm_eps=eps,
        rope_theta=theta,
        tie_word_embeddings=(eok == 1),
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        dtype="bfloat16",
        bos_token_id=151643,
        eos_token_id=151645,
    )
    config.save_pretrained(out_dir)

    for fname in TOKENIZER_FILES:
        shutil.copy(hf_hub_download(tokenizer_repo, fname), out_dir / fname)
    print(f"done -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tokenizer-repo", default="FermionResearch/Neutrino-8B")
    args = ap.parse_args()
    convert(args.container, args.out_dir, args.tokenizer_repo)


if __name__ == "__main__":
    main()
