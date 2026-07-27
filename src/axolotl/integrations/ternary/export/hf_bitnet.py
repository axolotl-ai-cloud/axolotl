"""transformers-bitnet packing: uint8 `(out_features / 4, in_features)` + scalar scale."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from axolotl.utils.logging import get_logger

from ..quant import CODES_PER_BYTE
from ..swap import SwapManifest
from . import bake

LOG = get_logger(__name__)

LANE_SHIFTS: tuple[int, ...] = (0, 2, 4, 6)
QUANT_METHOD: str = "bitnet"
WEIGHT_SCALE_SUFFIX: str = "weight_scale"

# archs whose ternary layout `BitNetForCausalLM` reproduces one-to-one
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {"llama", "mistral", "qwen2", "qwen3"}
)


def pack_hf_bitnet(codes: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes into the `BitNetForCausalLM` uint8 layout.

    Four output rows share one byte: value `code + 1` at bit shifts `[0, 2, 4, 6]`.
    The four rows are `out_features // 4` apart, not adjacent — lane `i` of packed
    row `r` holds original row `i * out_features // 4 + r`.

    Args:
        codes: int8 codes of shape `(out_features, in_features)`; `out_features`
            must be divisible by 4.

    Returns:
        uint8 tensor of shape `(out_features // 4, in_features)`.

    Raises:
        ValueError: If `out_features` is not divisible by 4.
    """
    if codes.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(codes.shape)}")
    out_features, in_features = codes.shape
    if out_features % CODES_PER_BYTE:
        raise ValueError(
            f"hf_bitnet needs out_features divisible by {CODES_PER_BYTE}, got "
            f"{out_features}; BitLinear sizes its packed buffer as out_features // "
            f"{CODES_PER_BYTE}, so a padded tensor cannot be loaded back"
        )
    lanes = (
        (codes.reshape(-1).to(torch.int16) + 1)
        .to(torch.uint8)
        .reshape(CODES_PER_BYTE, out_features // CODES_PER_BYTE, in_features)
    )
    packed = lanes[0].clone()
    for lane, shift in enumerate(LANE_SHIFTS[1:], start=1):
        packed |= lanes[lane] << shift
    return packed


def unpack_hf_bitnet(packed: torch.Tensor, out_features: int) -> torch.Tensor:
    """Inverse of `pack_hf_bitnet`, returning int8 codes of shape `(out_features, in_features)`."""
    if packed.ndim != 2:
        raise ValueError(
            f"expected a 2D packed tensor, got shape {tuple(packed.shape)}"
        )
    if out_features != packed.shape[0] * CODES_PER_BYTE:
        raise ValueError(
            f"packed tensor of shape {tuple(packed.shape)} holds "
            f"{packed.shape[0] * CODES_PER_BYTE} rows, not {out_features}"
        )
    lanes = torch.stack([(packed >> shift) & 3 for shift in LANE_SHIFTS])
    return lanes.reshape(out_features, packed.shape[1]).to(torch.int8) - 1


def quantization_config() -> dict:
    """Return the `quantization_config` block stamped into the exported `config.json`."""
    return {
        "quant_method": QUANT_METHOD,
        "linear_class": "bitlinear",
        "quantization_mode": "offline",
        "use_rms_norm": False,
        "rms_norm_eps": 1e-6,
        "modules_to_not_convert": None,
    }


def export_hf_bitnet(
    master_dir: str | Path, output_dir: str | Path, manifest: SwapManifest
) -> Path:
    """Write a packed transformers-bitnet checkpoint and return its directory.

    Raises:
        ValueError: If the manifest uses a scale mode the format cannot represent
            (anything but per-tensor), the architecture is unsupported, or the
            master still holds latent weights.
        KeyError: If a manifest entry has no matching weight in the master.
    """
    master_dir = Path(master_dir)
    output = Path(output_dir)
    _reject_unsupported(manifest)
    if output.resolve() == master_dir.resolve():
        raise ValueError(
            "hf_bitnet must be written beside the master, not over it; the packed "
            "weights are not re-quantizable back into a bf16 master"
        )

    entries = {f"{entry.name}.weight": entry for entry in manifest.entries}
    remaining = set(entries)
    shards = bake.shard_paths(master_dir)
    bake.copy_aux_files(
        master_dir,
        output,
        skip={path.name for path in shards}
        | {bake.SAFETENSORS_INDEX_NAME, bake.TORCH_INDEX_NAME},
    )

    weight_map: dict[str, str] = {}
    total_size = 0
    for path in shards:
        tensors, metadata = bake.load_shard(path)
        packed: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            entry = entries.get(key)
            if entry is None:
                packed[key] = tensor
                continue
            try:
                codes, scale = bake.derive_codes_and_scale(tensor)
            except ValueError as exc:
                raise ValueError(
                    f"{entry.name} still holds latent weights ({exc}); bake the master "
                    "before packing it"
                ) from exc
            packed[key] = pack_hf_bitnet(codes)
            # BitLinear dequantizes by dividing, so the companion holds 1 / absmean
            packed[f"{entry.name}.{WEIGHT_SCALE_SUFFIX}"] = scale.reciprocal().reshape(
                1
            )
            remaining.discard(key)
        shard_name = Path(path).name
        for key, tensor in packed.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        bake.save_shard(packed, output / shard_name, metadata)
    if remaining:
        raise KeyError(
            f"{len(remaining)} manifest weights are missing from the master in "
            f"{master_dir}: {sorted(remaining)[:5]}"
        )

    if (master_dir / bake.SAFETENSORS_INDEX_NAME).is_file():
        (output / bake.SAFETENSORS_INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_config(output, manifest)
    manifest.save(output)
    LOG.info(f"ternary: wrote {len(entries)} packed hf_bitnet tensors to {output}")
    return output


def _reject_unsupported(manifest: SwapManifest) -> None:
    if manifest.group_size is not None or manifest.weight_scale == "group":
        raise ValueError(
            "hf_bitnet stores one scalar weight_scale per tensor and cannot represent "
            f"ternary.weight_scale: {manifest.weight_scale} "
            f"(group_size={manifest.group_size}); export master_bf16 instead"
        )
    if manifest.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"hf_bitnet export does not support model_type {manifest.model_type!r} "
            f"(supported: {sorted(SUPPORTED_MODEL_TYPES)})"
        )


def _write_config(output_dir: Path, manifest: SwapManifest) -> None:
    path = output_dir / bake.CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {bake.CONFIG_FILENAME} in {output_dir}")
    config = json.loads(path.read_text(encoding="utf-8"))
    quant_config = quantization_config()
    quant_config["modules_to_not_convert"] = _modules_to_not_convert(manifest)
    config["quantization_config"] = quant_config
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bake.write_quantizer_metadata(output_dir, manifest, artifact_format="hf_bitnet")


def _modules_to_not_convert(manifest: SwapManifest) -> list[str]:
    # transformers prefix-matches these patterns, so an unanchored `...mlp.gate`
    # would also spare `...mlp.gate_proj`
    return [f"{re.escape(name)}$" for name in sorted(set(manifest.kept_fp))]


__all__ = [
    "LANE_SHIFTS",
    "QUANT_METHOD",
    "SUPPORTED_MODEL_TYPES",
    "WEIGHT_SCALE_SUFFIX",
    "export_hf_bitnet",
    "pack_hf_bitnet",
    "quantization_config",
    "unpack_hf_bitnet",
]
