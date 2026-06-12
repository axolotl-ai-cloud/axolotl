"""Convert the RedHatAI compressed-tensors NVFP4 DiffusionGemma checkpoint to the
fused (transformers >= 5.11) ``DiffusionGemmaForBlockDiffusion`` layout in bf16.

The published NVFP4 checkpoint was quantized from the pre-fusion layout
(transformers 5.8.0.dev0): experts are stored as *per-expert* compressed-tensors
Linears (``experts.0.gate_proj.weight_packed`` …) and the MLP is NVFP4-packed. The
released model class instead uses *fused 3D* experts (``experts.gate_up_proj`` /
``experts.down_proj``). This script:

  1. Decompresses every ``*.weight_packed`` module (NVFP4 -> bf16).
  2. Fuses per-expert ``gate_proj``/``up_proj``/``down_proj`` into the 3D tensors
     ``experts.gate_up_proj`` [E, 2I, H] and ``experts.down_proj`` [E, H, I].
  3. Keeps all unquantized tensors (attention, router, norms, vision tower) as-is.
  4. Writes a fused bf16 checkpoint (encoder text layers stay tied to the decoder).

The result loads cleanly into ``DiffusionGemmaForBlockDiffusion`` and can then be
4-bit (NVFP4) QLoRA-trained with axolotl's ``quantize_moe_experts`` path.

Usage:
    python scripts/convert_diffusiongemma_nvfp4_to_fused.py \
        --src RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
        --dst ./outputs/diffusiongemma-26B-A4B-it-fused-bf16
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from compressed_tensors.compressors import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationConfig
from huggingface_hub import save_torch_state_dict, snapshot_download
from safetensors import safe_open

PACKED_SUFFIX = ".weight_packed"
COMPRESSION_KEYS = (
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "weight_shape",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src", required=True, help="HF repo id or local dir of the NVFP4 checkpoint"
    )
    p.add_argument(
        "--dst", required=True, help="output dir for the fused bf16 checkpoint"
    )
    return p.parse_args()


def load_keymap(src_dir: Path):
    """Return {key: safe_open handle} across all shards."""
    shards = sorted(src_dir.glob("*.safetensors"))
    handles = [safe_open(str(s), framework="pt") for s in shards]
    key2h = {}
    for h in handles:
        for k in h.keys():
            key2h[k] = h
    return key2h


def main():
    args = parse_args()
    src_dir = Path(args.src)
    if not src_dir.exists():
        src_dir = Path(snapshot_download(args.src))
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(src_dir / "config.json"))
    scheme = QuantizationConfig.model_validate(
        cfg["quantization_config"]
    ).config_groups["group_0"]

    key2h = load_keymap(src_dir)
    keys = list(key2h.keys())

    # Group keys by their owning module (strip the trailing component).
    modules = defaultdict(dict)
    plain = {}
    for k in keys:
        base, _, leaf = k.rpartition(".")
        if leaf == "weight_packed" or leaf in COMPRESSION_KEYS:
            modules[base][leaf] = k
        else:
            plain[k] = k

    out: dict[str, torch.Tensor] = {}

    # 1. Decompress every packed module -> bf16 weight.
    n_decomp = 0
    for base, leaves in modules.items():
        if "weight_packed" not in leaves:
            continue
        sd = {leaf: key2h[k].get_tensor(k) for leaf, k in leaves.items()}
        dec = NVFP4PackedCompressor.decompress(sd, scheme)
        out[f"{base}.weight"] = dec["weight"].to(torch.bfloat16)
        n_decomp += 1
    print(f"decompressed {n_decomp} NVFP4 modules")

    # 2. Keep plain (unquantized) tensors; cast floats to bf16, leave int tensors alone.
    for k in plain:
        t = key2h[k].get_tensor(k)
        out[k] = t.to(torch.bfloat16) if t.is_floating_point() else t

    # 3. Fuse per-expert experts into 3D tensors, per (encoder/decoder, layer).
    #    Collect decompressed experts.{e}.{gate,up,down}_proj.weight -> stack.
    expert_re = re.compile(
        r"^(?P<pfx>.*\.experts)\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )
    grouped = defaultdict(lambda: defaultdict(dict))  # pfx -> e -> {proj: tensor}
    to_drop = []
    for k in list(out.keys()):
        m = expert_re.match(k)
        if m:
            grouped[m["pfx"]][int(m["e"])][m["proj"]] = out[k]
            to_drop.append(k)
    for k in to_drop:
        del out[k]

    for pfx, experts in grouped.items():
        E = max(experts) + 1
        gate_up, down = [], []
        for e in range(E):
            g = experts[e]["gate_proj"]  # [I, H]
            u = experts[e]["up_proj"]  # [I, H]
            d = experts[e]["down_proj"]  # [H, I]
            gate_up.append(torch.cat([g, u], dim=0))  # [2I, H]
            down.append(d)
        out[f"{pfx}.gate_up_proj"] = torch.stack(
            gate_up, dim=0
        ).contiguous()  # [E, 2I, H]
        out[f"{pfx}.down_proj"] = torch.stack(down, dim=0).contiguous()  # [E, H, I]
    print(f"fused {len(grouped)} expert blocks")

    # 4. Write config without quantization_config (now bf16) + sidecar files.
    cfg.pop("quantization_config", None)
    cfg["dtype"] = "bfloat16"
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    for fname in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "tokenizer.model",
    ):
        sp = src_dir / fname
        if sp.exists():
            shutil.copy(sp, dst / fname)

    print(f"saving {len(out)} tensors to {dst}")
    save_torch_state_dict(out, str(dst), max_shard_size="5GB")
    print("DONE")


if __name__ == "__main__":
    main()
