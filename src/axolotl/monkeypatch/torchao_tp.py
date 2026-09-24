"""Native TorchAO NVFP4 tensor-parallel component slicing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeNVFP4TPShard:
    """Logical shard bounds for one frozen NVFP4 parameter."""

    name: str
    dim: int
    start: int
    end: int


def _nvfp4_cls():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        return None
    return NVFP4Tensor


def _require_native_nvfp4(tensor) -> None:
    nvfp4_cls = _nvfp4_cls()
    if nvfp4_cls is None or not isinstance(tensor, nvfp4_cls):
        raise TypeError("Native tensor-parallel sharding requires an NVFP4Tensor")
    if tensor.ndim != 2 or not tensor.qdata.is_contiguous():
        raise ValueError(
            "Native NVFP4 tensor-parallel sharding requires contiguous rank-2 weights"
        )
    if tensor.block_size <= 0:
        raise ValueError(
            f"Native NVFP4 tensor has invalid block_size={tensor.block_size}"
        )


def _logical_scale_tensor(tensor):
    nvfp4_cls = _nvfp4_cls()
    if not tensor.is_swizzled_scales:
        return tensor
    from torchao.prototype.mx_formats.utils import from_blocked

    return nvfp4_cls(
        tensor.qdata,
        from_blocked(
            tensor.scale.flatten(),
            tensor.shape[0],
            tensor.shape[1] // tensor.block_size,
        ).contiguous(),
        tensor.block_size,
        tensor.orig_dtype,
        tensor.per_tensor_scale,
        tensor.act_per_tensor_scale,
        False,
        tensor.use_triton_kernel,
        tensor.act_quant_kwargs,
    )


def _repack_native_nvfp4_scales(tensor, swizzled):
    if not swizzled:
        return tensor
    from torchao.prototype.mx_formats.utils import (
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
        to_blocked,
    )

    scale_shape = hp_data_dims_to_swizzled_scale_dims_nvfp4(*tensor.shape)
    return _nvfp4_cls()(
        tensor.qdata,
        to_blocked(tensor.scale).reshape(scale_shape),
        tensor.block_size,
        tensor.orig_dtype,
        tensor.per_tensor_scale,
        tensor.act_per_tensor_scale,
        True,
        tensor.use_triton_kernel,
        tensor.act_quant_kwargs,
    )


def _owned_native_nvfp4(tensor):
    nvfp4_cls = _nvfp4_cls()
    return nvfp4_cls(
        tensor.qdata.clone(),
        tensor.scale.clone(),
        tensor.block_size,
        tensor.orig_dtype,
        None if tensor.per_tensor_scale is None else tensor.per_tensor_scale.clone(),
        None
        if tensor.act_per_tensor_scale is None
        else tensor.act_per_tensor_scale.clone(),
        tensor.is_swizzled_scales,
        tensor.use_triton_kernel,
        tensor.act_quant_kwargs,
    )


def native_nvfp4_tp_shard(
    tensor, dim: int, rank: int, world_size: int, *, name: str = "weight"
) -> NativeNVFP4TPShard:
    """Validate an upstream-style TP slice without changing the tensor."""
    _require_native_nvfp4(tensor)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid tensor-parallel rank {rank} for world size {world_size}"
        )
    if dim not in (-tensor.ndim, -tensor.ndim + 1, 0, 1):
        raise ValueError(f"Native NVFP4 TP dimension must be 0 or 1, got {dim}")
    dim %= tensor.ndim
    size = tensor.shape[dim]
    shard_size = (size + world_size - 1) // world_size
    start = rank * shard_size
    end = min(start + shard_size, size)
    if start >= end:
        raise ValueError(
            f"Native NVFP4 tensor-parallel sharding would create an empty shard for {name}"
        )

    alignment = tensor.block_size if dim == 1 else 1
    if start % alignment or (end != size and end % alignment):
        axis = "input" if dim == 1 else "output"
        raise ValueError(
            f"Native NVFP4 {axis}-axis TP shard for {name} must align to {alignment}; "
            f"got [{start}, {end}) of {size}"
        )
    return NativeNVFP4TPShard(name, dim, start, end)


def slice_native_nvfp4_tp(tensor, shard: NativeNVFP4TPShard):
    """Slice logically laid-out native components through TorchAO dispatch."""
    _require_native_nvfp4(tensor)
    logical = _logical_scale_tensor(tensor)
    sliced = torch.ops.aten.slice.Tensor(logical, shard.dim, shard.start, shard.end, 1)
    return _repack_native_nvfp4_scales(sliced, tensor.is_swizzled_scales)


def preflight_native_nvfp4_tp(
    named_parameters, parameter_dims: dict[str, int], rank: int, world_size: int
) -> tuple[NativeNVFP4TPShard, ...]:
    """Validate all native TP slices before a loader replaces any parameter."""
    parameters = dict(named_parameters)
    missing = sorted(set(parameter_dims) - set(parameters))
    if missing:
        raise ValueError(f"Native NVFP4 TP parameters are missing: {missing}")

    aliases: dict[int, NativeNVFP4TPShard] = {}
    shards = []
    for name, dim in parameter_dims.items():
        shard = native_nvfp4_tp_shard(
            parameters[name], dim, rank, world_size, name=name
        )
        previous = aliases.setdefault(id(parameters[name]), shard)
        if (previous.dim, previous.start, previous.end) != (
            shard.dim,
            shard.start,
            shard.end,
        ):
            raise ValueError(
                f"Native NVFP4 TP aliases disagree on shard placement: {name}"
            )
        shards.append(shard)
    return tuple(shards)


def materialize_native_nvfp4_tp(
    named_parameters, parameter_dims: dict[str, int], rank: int, world_size: int
) -> dict[str, torch.Tensor]:
    """Preflight and materialize local NVFP4 shards without mutating their owner modules."""
    parameters = dict(named_parameters)
    shards = preflight_native_nvfp4_tp(
        parameters.items(), parameter_dims, rank, world_size
    )
    by_identity = {}
    result = {}
    for shard in shards:
        parameter = parameters[shard.name]
        key = (id(parameter), shard.dim, shard.start, shard.end)
        if key not in by_identity:
            by_identity[key] = _owned_native_nvfp4(
                slice_native_nvfp4_tp(parameter, shard)
            )
        result[shard.name] = by_identity[key]
    return result
