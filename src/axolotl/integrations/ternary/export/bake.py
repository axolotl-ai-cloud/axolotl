"""Latent weights → exact `{-s16, 0, +s16}` bf16 master."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import axolotl
from axolotl.utils.logging import get_logger

from .. import quant
from ..args import LEARNABLE_SCALE_MODES
from ..swap import SwapManifest

LOG = get_logger(__name__)

CONFIG_FILENAME: str = "config.json"
QUANTIZER_METADATA_KEY: str = "ternary_quantizer"

SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

SAFETENSORS_NAME: str = "model.safetensors"
SAFETENSORS_INDEX_NAME: str = "model.safetensors.index.json"
TORCH_WEIGHTS_NAME: str = "pytorch_model.bin"
TORCH_INDEX_NAME: str = "pytorch_model.bin.index.json"


def bake_weight(
    weight: torch.Tensor,
    group_size: int | None = None,
    weight_scale: str = "absmean",
) -> torch.Tensor:
    """Return the latent weight replaced by `codes * s16`, same shape and dtype.

    Baking an already-baked weight is a no-op: re-running the quantizer would
    re-derive the scale from the ternary values themselves (absmean shrinks by the
    non-zero fraction) and shift every magnitude, so the tensor is returned as is.
    Reading a per-row or five-value master through the per-tensor grid is the same
    corruption, so `weight_scale` selects the grid the tensor is read on.

    Raises:
        ValueError: If the tensor is still a latent under a learned scale mode. The
            grid then lives in a `.scale` parameter that only the module has, so
            falling back to the absmean statistic here would quantize the checkpoint
            onto a scale the model never trained with.
    """
    if weight_scale == "trit_planes":
        if quant.baked_trit_plane_codes_and_scales(weight) is not None:
            return weight
        _reject_unbakeable_latent(weight, weight_scale)
        first, second = quant.trit_plane_absmean_scales(weight)
        return quant.fake_quant_weight_trit_planes(weight.detach(), 1.0, first, second)
    if weight_scale == "dual":
        if quant.baked_dual_codes_and_scales(weight) is not None:
            return weight
        _reject_unbakeable_latent(weight, weight_scale)
        low, high = quant.dual_absmean_scales(weight)
        return quant.fake_quant_weight_dual(weight.detach(), 1.0, low, high)
    group_size = _grid_group_size(weight, group_size, weight_scale)
    if quant.baked_codes_and_scale(weight, group_size) is not None:
        return weight
    _reject_unbakeable_latent(weight, weight_scale)
    return quant.fake_quant_weight(weight.detach(), 1.0, group_size)


def _reject_unbakeable_latent(weight: torch.Tensor, weight_scale: str) -> None:
    if weight_scale not in LEARNABLE_SCALE_MODES:
        return
    raise ValueError(
        f"a {tuple(weight.shape)} weight_scale: {weight_scale} tensor reached the "
        "export baker as a latent, not on its grid. Its scale is a trained parameter "
        "that the checkpoint alone cannot supply, so baking it here would invent an "
        "absmean grid the model never used. The module hook `_post_training` bakes "
        "these at save time — a master missing that bake is the bug to chase"
    )


def _grid_group_size(
    weight: torch.Tensor, group_size: int | None, weight_scale: str
) -> int | None:
    """Group size the quantizer sees: a per-row scale is one group of `in_features`."""
    if weight_scale == "learnable_row":
        return int(weight.shape[-1])
    return group_size


def bake_state_dict(
    state_dict: dict[str, torch.Tensor], manifest: SwapManifest
) -> dict[str, torch.Tensor]:
    """Bake every manifest-listed weight in a state dict, leaving the rest untouched.

    Args:
        state_dict: Latent-weight state dict.
        manifest: The swap manifest naming the ternary modules and their scale mode.

    Returns:
        A new state dict whose ternary entries hold exactly `{-s16, 0, +s16}`.

    Raises:
        KeyError: If a manifest entry has no matching weight in the state dict.
    """
    baked = dict(state_dict)
    for entry in manifest.entries:
        key = f"{entry.name}.weight"
        if key not in baked:
            raise KeyError(
                f"{key} is listed in the swap manifest but not in the state dict"
            )
        baked[key] = bake_weight(baked[key], entry.group_size, entry.weight_scale)
    return baked


def derive_codes_and_scale(
    weight: torch.Tensor,
    group_size: int | None = None,
    weight_scale: str = "absmean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the exact codes and `s16` scale from an already-baked master weight.

    Exact by construction: a baked tensor has at most three distinct magnitudes per
    scale unit, so every packer derives identical codes without re-running the
    quantizer.

    Returns:
        `(codes, scale)` — int8 codes shaped like `weight`, and the fp32 `s16` scale
        (0-dim per tensor, or one per group / row). Under `dual` the codes are the
        five-state `{-2..2}` grid and the scale is `(out_features, 2)` holding
        `(s_lo, s_hi)` per row; under `trit_planes` the codes are the two trit planes
        and the scale holds `(s1, s2)` per row. An all-zero tensor yields zero codes and the
        `SCALE_EPS` floor so downstream reciprocals stay finite.

    Raises:
        ValueError: If `weight` is not baked on `weight_scale`'s grid.
    """
    if weight_scale == "trit_planes":
        free_sum = quant.baked_trit_plane_codes_and_scales(weight)
        if free_sum is None:
            raise ValueError(
                f"tensor of shape {tuple(weight.shape)} is not baked: no per-row "
                "(s1, s2) reproduces its values on the nine-value free-sum grid"
            )
        planes, first, second = free_sum
        return planes, torch.cat([first, second], dim=-1)
    if weight_scale == "dual":
        dual = quant.baked_dual_codes_and_scales(weight)
        if dual is None:
            raise ValueError(
                f"tensor of shape {tuple(weight.shape)} is not baked: its rows are not "
                "exactly {-s_hi, -s_lo, 0, +s_lo, +s_hi}"
            )
        codes, low, high = dual
        return codes, torch.cat([low, high], dim=-1)
    recovered = quant.baked_codes_and_scale(
        weight, _grid_group_size(weight, group_size, weight_scale)
    )
    if recovered is None:
        raise ValueError(
            f"tensor of shape {tuple(weight.shape)} is not baked: its values are not "
            "exactly {-s, 0, +s}"
        )
    return recovered


def dequantize_derived(
    codes: torch.Tensor,
    scale: torch.Tensor,
    weight_scale: str = "absmean",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct the weight a `derive_codes_and_scale` pair stands for."""
    if weight_scale == "trit_planes":
        return quant.dequantize_trit_plane_codes(
            codes, scale[:, :1], scale[:, 1:], dtype
        )
    if weight_scale == "dual":
        return quant.dequantize_dual_codes(codes, scale[:, :1], scale[:, 1:], dtype)
    return quant.dequantize_codes(codes, scale, dtype)


def load_master(master_dir: str | Path) -> tuple[dict[str, torch.Tensor], SwapManifest]:
    """Load a baked master checkpoint and its swap manifest from `master_dir`."""
    return load_tensors(master_dir), SwapManifest.load(master_dir)


def load_tensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load every tensor of a (possibly sharded) checkpoint directory."""
    tensors: dict[str, torch.Tensor] = {}
    for path in shard_paths(model_dir):
        tensors.update(load_shard(path)[0])
    return tensors


def shard_paths(model_dir: str | Path) -> list[Path]:
    """Return the weight shards of a checkpoint directory, in index order.

    Raises:
        FileNotFoundError: If the directory holds no recognizable checkpoint.
    """
    directory = Path(model_dir)
    for index_name, single_name in (
        (SAFETENSORS_INDEX_NAME, SAFETENSORS_NAME),
        (TORCH_INDEX_NAME, TORCH_WEIGHTS_NAME),
    ):
        index = directory / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            return [directory / name for name in dict.fromkeys(weight_map.values())]
        if (directory / single_name).is_file():
            return [directory / single_name]
    raise FileNotFoundError(f"no model weights found in {model_dir}")


def tensor_dtypes(model_dir: str | Path) -> dict[str, torch.dtype]:
    """Return each tensor's storage dtype, read from the shard headers only.

    A checkpoint mixes dtypes — the ternary masters are saved in their module's dtype
    while norms and embeddings stay fp32 — so `config.dtype` cannot answer this.
    Non-safetensors shards are skipped rather than loaded.
    """
    dtypes: dict[str, torch.dtype] = {}
    for path in shard_paths(model_dir):
        if path.suffix != ".safetensors":
            continue
        with safe_open(path, framework="pt") as shard:
            for key in shard.keys():  # noqa: SIM118 — safetensors handle, not a mapping
                resolved = SAFETENSORS_DTYPES.get(shard.get_slice(key).get_dtype())
                if resolved is not None:
                    dtypes[key] = resolved
    return dtypes


def load_shard(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load one checkpoint shard and its file-level metadata."""
    path = Path(path)
    if path.suffix != ".safetensors":
        return torch.load(path, map_location="cpu", weights_only=True), {}
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as shard:
        metadata = shard.metadata() or {}
        for key in shard.keys():  # noqa: SIM118 — safetensors handle, not a mapping
            tensors[key] = shard.get_tensor(key)
    return tensors, metadata


def save_shard(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write one checkpoint shard, preserving the safetensors metadata header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".safetensors":
        torch.save(tensors, path)
        return
    header = dict(metadata or {})
    header.setdefault("format", "pt")
    save_file({key: value.contiguous() for key, value in tensors.items()}, path, header)


def copy_aux_files(
    src_dir: str | Path, dst_dir: str | Path, skip: Iterable[str] = ()
) -> None:
    """Copy every non-weight file (config, tokenizer, ...) from `src_dir` to `dst_dir`."""
    skipped = set(skip)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(src_dir).iterdir()):
        if path.is_file() and path.name not in skipped:
            shutil.copy2(path, dst / path.name)


def bake_directory(
    master_dir: str | Path,
    output_dir: str | Path | None = None,
    manifest: SwapManifest | None = None,
) -> Path:
    """Bake a saved checkpoint in place (or into `output_dir`) and stamp its metadata.

    Idempotent: a directory whose ternary tensors already hold `{-s16, 0, +s16}` is
    rewritten byte-identically. Sharding is preserved.

    Args:
        master_dir: Directory holding the saved checkpoint and its swap manifest.
        output_dir: Where to write; defaults to `master_dir` (in-place).
        manifest: Manifest to bake against; read from `master_dir` when `None`.

    Returns:
        The baked directory.

    Raises:
        FileNotFoundError: If the checkpoint or its manifest is missing.
        KeyError: If a manifest entry has no matching weight in the checkpoint.
    """
    master_dir = Path(master_dir)
    output = Path(output_dir) if output_dir is not None else master_dir
    manifest = manifest if manifest is not None else SwapManifest.load(master_dir)

    shards = shard_paths(master_dir)
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != master_dir.resolve():
        copy_aux_files(master_dir, output, skip={path.name for path in shards})

    grids = {
        f"{entry.name}.weight": (entry.group_size, entry.weight_scale)
        for entry in manifest.entries
    }
    remaining = set(grids)
    for path in shards:
        tensors, metadata = load_shard(path)
        for key in tensors.keys() & remaining:
            tensors[key] = bake_weight(tensors[key], *grids[key])
            remaining.discard(key)
        save_shard(tensors, output / path.name, metadata)
    if remaining:
        raise KeyError(
            f"{len(remaining)} manifest weights are missing from the checkpoint in "
            f"{master_dir}: {sorted(remaining)[:5]}"
        )

    manifest.save(output)
    write_quantizer_metadata(output, manifest)
    LOG.info(f"ternary: baked {len(grids)} weights into {output}")
    return output


def write_quantizer_metadata(
    model_dir: str | Path, manifest: SwapManifest, artifact_format: str = "master_bf16"
) -> bool:
    """Record the quantizer identity in the checkpoint's `config.json`.

    Export and the parity gates read this to tell a baked ternary checkpoint from an
    ordinary bf16 one; `PretrainedConfig` keeps the extra key untouched on load.

    Returns:
        Whether the directory had a `config.json` to stamp.
    """
    path = Path(model_dir) / CONFIG_FILENAME
    if not path.is_file():
        return False
    config = json.loads(path.read_text(encoding="utf-8"))
    config[QUANTIZER_METADATA_KEY] = {
        **manifest.quantizer,
        "format": artifact_format,
        "weight_scale": manifest.weight_scale,
        "group_size": manifest.group_size,
        "activation_bits": manifest.activation_bits,
        "modules": len(manifest.entries),
        "axolotl_version": axolotl.__version__,
    }
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return True


__all__ = [
    "CONFIG_FILENAME",
    "QUANTIZER_METADATA_KEY",
    "SAFETENSORS_DTYPES",
    "SAFETENSORS_INDEX_NAME",
    "SAFETENSORS_NAME",
    "bake_directory",
    "bake_state_dict",
    "bake_weight",
    "copy_aux_files",
    "dequantize_derived",
    "derive_codes_and_scale",
    "load_master",
    "load_shard",
    "load_tensors",
    "save_shard",
    "shard_paths",
    "tensor_dtypes",
    "write_quantizer_metadata",
]
