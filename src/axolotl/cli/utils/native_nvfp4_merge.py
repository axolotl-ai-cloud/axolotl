"""Format-preserving shard merge helpers for native TorchAO NVFP4 checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable

import torch

from axolotl.monkeypatch.torchao_nvfp4_merge import (
    NativeNVFP4Recipe,
    capture_native_nvfp4_recipe,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
DenseMerge = Callable[[torch.Tensor, str], tuple[torch.Tensor, bool]]


def _flat_prefix(name: str) -> str:
    parent, separator, leaf = name.rpartition(".")
    return f"{parent + separator if parent else ''}_{leaf}_"


def has_native_nvfp4_weights(metadata: dict[str, str]) -> bool:
    """Whether safetensors metadata contains native TorchAO NVFP4 weights."""
    return bool(_native_weight_names(metadata))


def _native_weight_names(metadata: dict[str, str]) -> list[str]:
    names = json.loads(metadata.get("tensor_names", "[]"))
    result = []
    for name in names:
        encoded = metadata.get(name)
        if encoded is None:
            continue
        try:
            if json.loads(encoded).get("_type") == "NVFP4Tensor":
                result.append(name)
        except json.JSONDecodeError:
            continue
    return result


def _metadata_for_name(metadata: dict[str, str], name: str) -> dict[str, str]:
    return {name: metadata[name], "tensor_names": json.dumps([name])}


def _recipe_on_device(
    recipe: NativeNVFP4Recipe, device: torch.device
) -> NativeNVFP4Recipe:
    return NativeNVFP4Recipe(
        block_size=recipe.block_size,
        orig_dtype=recipe.orig_dtype,
        per_tensor_scale=None
        if recipe.per_tensor_scale is None
        else recipe.per_tensor_scale.to(device),
        act_per_tensor_scale=None
        if recipe.act_per_tensor_scale is None
        else recipe.act_per_tensor_scale.to(device),
        is_swizzled_scales=recipe.is_swizzled_scales,
        use_triton_kernel=recipe.use_triton_kernel,
        act_quant_kwargs=recipe.act_quant_kwargs,
    )


def _unflatten_native_weight(shard_tensors, metadata, name):
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    if "." in name:
        native, _ = unflatten_tensor_state_dict(
            shard_tensors, _metadata_for_name(metadata, name)
        )
        return native.get(name)
    sentinel = "__axolotl_root__"
    prefixed = {
        sentinel + "." + key if key.startswith("_") else key: value
        for key, value in shard_tensors.items()
    }
    entry = {
        sentinel + "." + name: metadata[name],
        "tensor_names": json.dumps([sentinel + "." + name]),
    }
    native, _ = unflatten_tensor_state_dict(prefixed, entry)
    return native.get(sentinel + "." + name)


def _flatten_native_weight(name, snapped):
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )

    if "." in name:
        return flatten_tensor_state_dict({name: snapped})
    sentinel = "__axolotl_root__"
    flattened, metadata = flatten_tensor_state_dict({sentinel + "." + name: snapped})
    prefix = sentinel + "._"
    flattened = {
        key[len(sentinel) + 1 :] if key.startswith(prefix) else key: value
        for key, value in flattened.items()
    }
    return flattened, {
        name: metadata[sentinel + "." + name],
        "tensor_names": json.dumps([name]),
    }


def merge_native_nvfp4_shard(
    shard_tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    merge_dense_weight: DenseMerge,
    *,
    quantization_device: str | torch.device | None = None,
    dequant: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, str], int]:
    """Merge native NVFP4 logical weights, optionally emitting dense weights.

    ``merge_dense_weight`` receives a dequantized native weight and its logical
    state-dict key, returning the effective dense weight and whether it had LoRA.
    It lets the top-level merger retain its existing PEFT key mapping and delta
    construction without teaching this format helper about adapter layouts.
    """
    names = _native_weight_names(metadata)
    if not names:
        return shard_tensors, metadata, 0

    result = dict(shard_tensors)
    result_metadata = dict(metadata)
    merged = 0
    for name in names:
        try:
            entry = json.loads(metadata[name])
        except json.JSONDecodeError:
            continue
        prefix = _flat_prefix(name)
        required = [prefix + suffix for suffix in entry.get("_tensor_data_names", ())]
        if any(key not in shard_tensors for key in required):
            LOG.warning(
                "NVFP4 MERGE WARNING: native weight %s has split or incomplete "
                "TorchAO components in this shard; preserving it unchanged",
                name,
            )
            continue
        weight = _unflatten_native_weight(shard_tensors, metadata, name)
        if type(weight).__name__ != "NVFP4Tensor":
            LOG.warning(
                "NVFP4 MERGE WARNING: native weight %s could not be reconstructed; "
                "preserving it unchanged",
                name,
            )
            continue
        effective, did_merge = merge_dense_weight(weight.dequantize(), name)
        if not did_merge and not dequant:
            continue
        if dequant:
            result = {
                key: value
                for key, value in result.items()
                if not key.startswith(prefix)
            }
            result[name] = effective.to(weight.orig_dtype)
            result_metadata[name] = json.dumps({"_type": "Tensor"})
            merged += int(did_merge)
            continue
        recipe = capture_native_nvfp4_recipe(weight)
        device = torch.device(quantization_device or effective.device)
        if recipe.use_triton_kernel and device.type != "cuda":
            LOG.warning(
                "NVFP4 MERGE WARNING: native weight %s requires CUDA for its Triton "
                "encoder; preserving it unchanged",
                name,
            )
            continue
        snapped = _recipe_on_device(recipe, device).quantize(effective.to(device))
        result = {
            key: value for key, value in result.items() if not key.startswith(prefix)
        }
        flattened, replacement_metadata = _flatten_native_weight(name, snapped)
        result.update(flattened)
        result_metadata[name] = replacement_metadata[name]
        merged += 1
    return result, result_metadata, merged
