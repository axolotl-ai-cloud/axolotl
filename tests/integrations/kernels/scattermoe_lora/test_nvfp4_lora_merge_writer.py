# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end NVFP4 expert-LoRA merge writer test (step 3).

Synthesizes a tiny dsv4-scheme NVFP4 base checkpoint and a tiny expert-LoRA adapter on disk,
runs ``write_merged_nvfp4_checkpoint``, then reloads the OUTPUT through the SAME loader fusion
(``_build_expert_nvfp4`` -> NVFP4Tensor -> dequantize) and asserts the merged dequant matches an
independent reconstruction (base dequant + reconstructed LoRA delta) within NVFP4 requant
tolerance. Also pins the structural invariants: identical index key set, byte-identical
non-expert passthrough, and gate/up sharing one weight_scale_2.

The writer + its two sibling deps are loaded BY FILE PATH so the scattermoe_lora package
__init__ (which imports triton) is NOT executed; gated on torchao + safetensors, NOT CUDA."""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch

torchao = pytest.importorskip("torchao", reason="torchao required for NVFP4 requant")
pytest.importorskip("safetensors", reason="safetensors required for checkpoint IO")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: E402
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)


def _load_by_path(mod_name: str, filename: str):
    """Load a scattermoe_lora module by file path (avoids the triton-importing package __init__)."""
    import axolotl

    path = os.path.join(
        os.path.dirname(axolotl.__file__),
        "integrations",
        "kernels",
        "libs",
        "scattermoe_lora",
        filename,
    )
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge_mod = _load_by_path("_writer_test_merge", "nvfp4_lora_merge.py")
loading_mod = _load_by_path("_writer_test_loading", "nvfp4_moe_loading.py")
writer_mod = _load_by_path("_writer_test_writer", "nvfp4_lora_merge_writer.py")
# Inject the file-path-loaded siblings so the writer never triggers the package __init__.
writer_mod._SIBLINGS = {
    "nvfp4_lora_merge": merge_mod,
    "nvfp4_moe_loading": loading_mod,
}


DSV4_BASE_FMT = "layers.{layer}.ffn.experts.{e}.{proj}"
H = 32  # hidden
I = 16  # intermediate
E = 2  # experts
BLOCK = 16


def _quantize_proj(w: torch.Tensor, pts: torch.Tensor):
    t = NVFP4Tensor.to_nvfp4(w, block_size=BLOCK, per_tensor_scale=pts)
    # clone: gate/up share one pts object; safetensors rejects keys aliasing the same storage.
    return t.qdata, t.scale, pts.to(torch.float32).clone()


def _make_base_checkpoint(base_dir: str):
    """Write a tiny dsv4 NVFP4 base: 1 MoE layer, E=2 experts, plus 2 non-expert passthrough tensors.

    Returns the bf16 source weights and shared gate/up per-tensor scales for independent
    reconstruction in the test."""
    os.makedirs(base_dir, exist_ok=True)
    g = torch.Generator().manual_seed(7)

    src = {}  # (e, proj) -> bf16 weight
    pts = {}  # (e, proj) -> per_tensor_scale (gate/up share)
    tensors: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    shard = "model-00001-of-00001.safetensors"

    for e in range(E):
        w1 = torch.randn(I, H, generator=g, dtype=torch.bfloat16)  # gate
        w3 = torch.randn(I, H, generator=g, dtype=torch.bfloat16)  # up
        w2 = torch.randn(H, I, generator=g, dtype=torch.bfloat16)  # down
        # gate (w1) and up (w3) MUST share one weight_scale_2 (the source invariant).
        shared_pts = per_tensor_amax_to_scale(torch.cat([w1, w3], dim=0).abs().max())
        down_pts = per_tensor_amax_to_scale(w2.abs().max())
        for proj, w, p in (
            ("w1", w1, shared_pts),
            ("w3", w3, shared_pts),
            ("w2", w2, down_pts),
        ):
            qd, sc, p2 = _quantize_proj(w, p)
            base = DSV4_BASE_FMT.format(layer=0, e=e, proj=proj)
            tensors[f"{base}.weight"] = qd
            tensors[f"{base}.weight_scale"] = sc
            tensors[f"{base}.weight_scale_2"] = p2
            src[(e, proj)] = w
            pts[(e, proj)] = p

    # Non-expert passthrough tensors.
    tensors["model.embed_tokens.weight"] = torch.randn(
        8, H, generator=g, dtype=torch.bfloat16
    )
    tensors["layers.0.input_layernorm.weight"] = torch.randn(
        H, generator=g, dtype=torch.bfloat16
    )

    for key in tensors:
        weight_map[key] = shard
    save_file(tensors, os.path.join(base_dir, shard), metadata={"format": "pt"})

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    with open(os.path.join(base_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump({"n_routed_experts": E, "model_type": "dsv4_test"}, f)

    return src, pts


def _make_adapter(adapter_dir: str, r: int, alpha: float):
    """Write a tiny adapter: gate_up_proj (fused out=2I, in=H) + down_proj (out=H, in=I).

    Returns the raw PEFT lora_A/lora_B per proj for independent delta reconstruction."""
    os.makedirs(adapter_dir, exist_ok=True)
    g = torch.Generator().manual_seed(13)

    gup_out, gup_in = 2 * I, H
    dn_out, dn_in = H, I
    ab = {
        "gate_up_proj": (
            torch.randn(r * E, gup_in, generator=g, dtype=torch.bfloat16),
            torch.randn(gup_out, r * E, generator=g, dtype=torch.bfloat16),
        ),
        "down_proj": (
            torch.randn(r * E, dn_in, generator=g, dtype=torch.bfloat16),
            torch.randn(dn_out, r * E, generator=g, dtype=torch.bfloat16),
        ),
    }

    sd = {}
    for proj, (A, B) in ab.items():
        prefix = f"base_model.model.model.layers.0.mlp.experts.{proj}"
        sd[f"{prefix}.lora_A.weight"] = A
        sd[f"{prefix}.lora_B.weight"] = B
    save_file(sd, os.path.join(adapter_dir, "adapter_model.safetensors"))

    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump({"r": r, "lora_alpha": alpha, "use_rslora": False}, f)

    return ab


def _reload_fused_dequant(repo, projs):
    """Reload the output exactly as the loaders do and dequantize the fused expert block."""
    wmap = loading_mod._load_index(repo)
    qd, sc, pts = loading_mod._build_expert_nvfp4(
        repo, wmap, DSV4_BASE_FMT, 0, projs, E, "cpu"
    )
    # Per-expert dequant from each expert's own scalar scale (fused[e] does not slice pts).
    pe = pts.reshape(-1)
    dq = []
    for e in range(E):
        t_e = NVFP4Tensor(qd[e], sc[e], BLOCK, torch.bfloat16, per_tensor_scale=pe[e])
        dq.append(t_e.dequantize(torch.bfloat16))
    return torch.stack(dq, 0)


def _expected_merged(src, pts, ab, projs, scaling):
    """Independent reconstruction: dequant the BASE fused NVFP4 + add reconstructed LoRA delta."""
    # Build the fused base dequant per expert from the source bf16 weights, re-quantized exactly
    # as _make_base_checkpoint did (so this is the same on-disk base the writer read).
    base_dq = []
    for e in range(E):
        parts = []
        for proj in projs:
            qd, sc, _ = _quantize_proj(src[(e, proj)], pts[(e, proj)])
            t = NVFP4Tensor(
                qd, sc, BLOCK, torch.bfloat16, per_tensor_scale=pts[(e, proj)]
            )
            parts.append(t.dequantize(torch.bfloat16))
        base_dq.append(torch.cat(parts, dim=0))
    base_dq = torch.stack(base_dq, 0)  # [E, sumOut, in]

    proj_name = "gate_up_proj" if len(projs) > 1 else "down_proj"
    A, B = ab[proj_name]
    rank = A.shape[0] // E
    sm_B = merge_mod._peft_lora_B_to_scattermoe(B, E, rank)
    delta = merge_mod.reconstruct_expert_delta(A, sm_B, scaling, E, rank)
    return (base_dq.float() + delta.float()).to(torch.bfloat16)


def test_write_merged_nvfp4_checkpoint_end_to_end(tmp_path):
    base_dir = str(tmp_path / "base")
    adapter_dir = str(tmp_path / "adapter")
    out_dir = str(tmp_path / "out")

    r, alpha = 4, 8.0
    scaling = alpha / r
    src, pts = _make_base_checkpoint(base_dir)
    ab = _make_adapter(adapter_dir, r, alpha)

    writer_mod.write_merged_nvfp4_checkpoint(base_dir, adapter_dir, out_dir)

    base_wmap = loading_mod._load_index(base_dir)
    out_wmap = loading_mod._load_index(out_dir)

    # (a) identical key set, no missing/extra.
    assert set(out_wmap.keys()) == set(base_wmap.keys())

    # (b) non-expert tensors byte-identical.
    base_f = safe_open(
        os.path.join(base_dir, "model-00001-of-00001.safetensors"), framework="pt"
    )
    out_f = safe_open(
        os.path.join(out_dir, "model-00001-of-00001.safetensors"), framework="pt"
    )
    for key in ("model.embed_tokens.weight", "layers.0.input_layernorm.weight"):
        assert torch.equal(base_f.get_tensor(key), out_f.get_tensor(key)), key

    # (c) gate (w1) and up (w3) share weight_scale_2 in the output.
    for e in range(E):
        w1_p2 = out_f.get_tensor(f"layers.0.ffn.experts.{e}.w1.weight_scale_2")
        w3_p2 = out_f.get_tensor(f"layers.0.ffn.experts.{e}.w3.weight_scale_2")
        assert torch.equal(w1_p2.float(), w3_p2.float()), e

    # End-to-end: reloaded merged dequant matches independent reconstruction within NVFP4 tol.
    gu_got = _reload_fused_dequant(out_dir, ("w1", "w3"))
    dn_got = _reload_fused_dequant(out_dir, ("w2",))
    gu_exp = _expected_merged(src, pts, ab, ("w1", "w3"), scaling)
    dn_exp = _expected_merged(src, pts, ab, ("w2",), scaling)

    for got, exp, name in ((gu_got, gu_exp, "gate_up"), (dn_got, dn_exp, "down")):
        for e in range(E):
            err = (got[e].float() - exp[e].float()).abs().max().item()
            amax = exp[e].float().abs().max().item()
            assert err <= 0.15 * amax, f"{name} expert {e}: err {err} amax {amax}"


def test_write_merged_nvfp4_checkpoint_copies_config(tmp_path):
    base_dir = str(tmp_path / "base")
    adapter_dir = str(tmp_path / "adapter")
    out_dir = str(tmp_path / "out")

    _make_base_checkpoint(base_dir)
    _make_adapter(adapter_dir, r=4, alpha=8.0)
    writer_mod.write_merged_nvfp4_checkpoint(base_dir, adapter_dir, out_dir)

    with open(os.path.join(out_dir, "config.json")) as f:
        assert json.load(f)["n_routed_experts"] == E
    assert os.path.isfile(os.path.join(out_dir, "model.safetensors.index.json"))


def test_write_merged_nvfp4_checkpoint_shards_by_size(tmp_path):
    """A tiny max_shard_bytes forces the writer to spill across multiple shards; the rebuilt index
    must span them, every referenced shard file must exist, the key set must be preserved, and the
    round-trip dequant must still match (the bounded-memory path produces an equivalent checkpoint)."""
    base_dir = str(tmp_path / "base")
    adapter_dir = str(tmp_path / "adapter")
    out_dir = str(tmp_path / "out")

    r, alpha = 4, 8.0
    scaling = alpha / r
    src, pts = _make_base_checkpoint(base_dir)
    ab = _make_adapter(adapter_dir, r, alpha)

    # Tiny budget so several tensors cannot share one shard (forces the flush/rename path).
    writer_mod.write_merged_nvfp4_checkpoint(
        base_dir, adapter_dir, out_dir, max_shard_bytes=256
    )

    base_wmap = loading_mod._load_index(base_dir)
    out_wmap = loading_mod._load_index(out_dir)

    shards = set(out_wmap.values())
    assert len(shards) > 1, "tiny budget should produce multiple shards"
    for shard in shards:
        assert os.path.isfile(os.path.join(out_dir, shard)), shard
    # No leftover temp shards from the flush-then-rename step.
    assert not any(name.startswith("_shard-") for name in os.listdir(out_dir))
    assert set(out_wmap.keys()) == set(base_wmap.keys())

    # Each expert's weight/weight_scale/weight_scale_2 triple must stay in ONE shard: the loader
    # reads the two scales from the shard of the .weight key, so a split triple breaks reload.
    for e in range(E):
        for proj in ("w1", "w3", "w2"):
            b = DSV4_BASE_FMT.format(layer=0, e=e, proj=proj)
            triple = {
                out_wmap[f"{b}.{s}"]
                for s in ("weight", "weight_scale", "weight_scale_2")
            }
            assert len(triple) == 1, f"{b} triple split across shards {triple}"

    gu_got = _reload_fused_dequant(out_dir, ("w1", "w3"))
    gu_exp = _expected_merged(src, pts, ab, ("w1", "w3"), scaling)
    for e in range(E):
        err = (gu_got[e].float() - gu_exp[e].float()).abs().max().item()
        amax = gu_exp[e].float().abs().max().item()
        assert err <= 0.15 * amax, f"expert {e}: err {err} amax {amax}"
