"""Native TorchAO NVFP4 tensor-parallel component slicing helpers."""

from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator

import torch

_DENSE_TP_DIMS = {"colwise": 0, "colwise_gather_output": 0, "rowwise": 1}


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


def _native_nvfp4_tp_shard_shape(
    shape, dim: int, rank: int, world_size: int, *, block_size: int, name: str
) -> NativeNVFP4TPShard:
    if len(shape) != 2:
        raise ValueError(
            "Native NVFP4 tensor-parallel sharding requires rank-2 weights"
        )
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid tensor-parallel rank {rank} for world size {world_size}"
        )
    if dim not in (-2, -1, 0, 1):
        raise ValueError(f"Native NVFP4 TP dimension must be 0 or 1, got {dim}")
    dim %= 2
    size = shape[dim]
    shard_size = (size + world_size - 1) // world_size
    start = rank * shard_size
    end = min(start + shard_size, size)
    if start >= end:
        raise ValueError(
            f"Native NVFP4 tensor-parallel sharding would create an empty shard for {name}"
        )
    alignment = block_size if dim == 1 else 1
    if start % alignment or (end != size and end % alignment):
        axis = "input" if dim == 1 else "output"
        raise ValueError(
            f"Native NVFP4 {axis}-axis TP shard for {name} must align to {alignment}; "
            f"got [{start}, {end}) of {size}"
        )
    return NativeNVFP4TPShard(name, dim, start, end)


def native_nvfp4_tp_shard(
    tensor, dim: int, rank: int, world_size: int, *, name: str = "weight"
) -> NativeNVFP4TPShard:
    """Validate an upstream-style TP slice without changing the tensor."""
    _require_native_nvfp4(tensor)
    return _native_nvfp4_tp_shard_shape(
        tensor.shape,
        dim,
        rank,
        world_size,
        block_size=tensor.block_size,
        name=name,
    )


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


class _NativeNVFP4ComponentPlacement:
    """Keep serialized components whole until their NVFP4 owner is rebuilt."""

    def __init__(self, device_mesh):
        self.device_mesh = device_mesh
        self.device = None
        self.dim = None
        self.plan = None

    def __deepcopy__(self, memo):
        return type(self)(self.device_mesh)

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        del tensor_idx, dtype
        self.device = device
        return param[...]

    def get_expected_sharded_shape(self, full_shape):
        return tuple(full_shape)

    def update_module_attributes(self, module):
        if self.plan == "colwise" and hasattr(module, "out_features"):
            module.out_features = module.weight.shape[0]
        elif self.dim == 1 and hasattr(module, "in_features"):
            module.in_features = module.weight.shape[1]


def _metadata_has_nvfp4(metadata) -> bool:
    if isinstance(metadata, dict):
        return metadata.get("_type") == "NVFP4Tensor" or any(
            _metadata_has_nvfp4(value) for value in metadata.values()
        )
    if isinstance(metadata, list):
        return any(_metadata_has_nvfp4(value) for value in metadata)
    return False


def _tp_mesh(device_mesh):
    if getattr(device_mesh, "ndim", 1) > 1:
        return device_mesh["tp"]
    return device_mesh


def _native_nvfp4_tp_plan(model, layer_name):
    import re

    generic_name = re.sub(
        r"\.\d+(\.|$)", lambda match: ".*" + match.group(1), layer_name
    )
    plan = model.tp_plan.get(generic_name)
    if plan is None and "." in generic_name:
        plan = model.tp_plan.get(generic_name.rsplit(".", 1)[0])
    if plan is None:
        return None
    if plan not in _DENSE_TP_DIMS:
        raise ValueError(
            f"Native NVFP4 TP only supports dense colwise or rowwise plans, got {plan!r} for {layer_name}"
        )
    return plan


def _native_nvfp4_tp_dim(model, layer_name):
    plan = _native_nvfp4_tp_plan(model, layer_name)
    return None if plan is None else _DENSE_TP_DIMS[plan]


def _native_nvfp4_weight_converter(
    base_converter, device_mesh, target_devices=None, target_devices_by_name=None
):
    from transformers.core_model_loading import WeightConverter

    class NativeNVFP4WeightConverter(WeightConverter):
        def __init__(
            self,
            source_patterns,
            target_patterns,
            operations,
            target_devices=target_devices,
            target_devices_by_name=target_devices_by_name,
        ):
            super().__init__(source_patterns, target_patterns, operations)
            self.distributed_operation = _NativeNVFP4ComponentPlacement(device_mesh)
            self.target_devices = {} if target_devices is None else target_devices
            self.target_devices_by_name = (
                {} if target_devices_by_name is None else target_devices_by_name
            )

        def __deepcopy__(self, memo):
            copied = type(self)(
                self.source_patterns,
                self.target_patterns,
                deepcopy(self.operations, memo),
                self.target_devices,
                self.target_devices_by_name,
            )
            memo[id(self)] = copied
            return copied

        def convert(self, layer_name, model=None, **kwargs):
            values = super().convert(layer_name, model=model, **kwargs)
            plan = _native_nvfp4_tp_plan(model, layer_name)
            dim = None if plan is None else _DENSE_TP_DIMS[plan]
            self.distributed_operation.dim = dim
            self.distributed_operation.plan = plan
            if dim is None:
                if self.distributed_operation.device is not None:
                    for name, value in values.items():
                        if isinstance(value, list):
                            if len(value) != 1:
                                raise ValueError(
                                    f"Native NVFP4 deserialization returned {len(value)} values for {name}"
                                )
                            value = value[0]
                        values[name] = value.to(self.distributed_operation.device)
                return values
            rank = self.distributed_operation.device_mesh.get_local_rank()
            world_size = self.distributed_operation.device_mesh.size()
            manifest = getattr(model, "_axolotl_native_nvfp4_tp_manifest", {})
            for name, value in values.items():
                if isinstance(value, list):
                    if len(value) != 1:
                        raise ValueError(
                            f"Native NVFP4 deserialization returned {len(value)} values for {name}"
                        )
                    value = value[0]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"Native NVFP4 deserialization returned {type(value).__name__} for {name}"
                    )
                shard = native_nvfp4_tp_shard(value, dim, rank, world_size, name=name)
                try:
                    expected = manifest[name]
                except KeyError as exc:
                    raise ValueError(
                        f"Native NVFP4 TP checkpoint parameter was not preflighted: {name}"
                    ) from exc
                if shard != expected:
                    raise ValueError(
                        f"Native NVFP4 TP checkpoint shape changed after preflight: {name}"
                    )
                value = _owned_native_nvfp4(slice_native_nvfp4_tp(value, shard))
                target = (
                    model.get_parameter(name)
                    if hasattr(model, "get_parameter")
                    else None
                )
                target_device = self.target_devices.get(id(target))
                if target_device is None:
                    target_device = self.target_devices_by_name.get(name)
                if target_device is None:
                    target_device = getattr(target, "device", None)
                if target_device is None or target_device.type == "meta":
                    target_device = self.distributed_operation.device
                if target_device is not None:
                    value = value.to(target_device)
                values[name] = value
            return values

    return NativeNVFP4WeightConverter(
        base_converter.source_patterns,
        base_converter.target_patterns,
        base_converter.operations,
    )


@contextmanager
def native_nvfp4_tp_checkpoint_loading(device_mesh) -> Iterator[None]:
    """Install a scoped TorchAO converter which shards after deserialization."""
    from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer

    device_mesh = _tp_mesh(device_mesh)
    original = TorchAoHfQuantizer.get_weight_conversions
    original_preprocess = TorchAoHfQuantizer._process_model_before_weight_loading
    from transformers import PreTrainedModel
    from transformers.core_model_loading import DtensorShardOperation

    distribute_owner = PreTrainedModel
    distribute_name = "maybe_distribute_model"
    had_own_distribute_model = distribute_name in distribute_owner.__dict__
    original_distribute_descriptor = inspect.getattr_static(
        distribute_owner, distribute_name
    )
    original_dtensor_init = DtensorShardOperation.__init__
    original_shard_tensor = DtensorShardOperation.shard_tensor
    native_dtensor_parameters = {}
    native_dtensor_names = {}
    native_dtensor_devices = {}
    native_dtensor_devices_by_name = {}
    state_dict_restorations = []

    def distribute_model(cls, model, *args, **kwargs):
        original_distribute_model = original_distribute_descriptor.__get__(None, cls)
        distributed = original_distribute_model(model, *args, **kwargs)
        native_names = getattr(distributed, "_axolotl_native_nvfp4_tp_manifest", {})
        for name, parameter in distributed.named_parameters(remove_duplicate=False):
            if name in native_names:
                native_dtensor_parameters[id(parameter)] = parameter
                native_dtensor_names[id(parameter)] = name
        if not hasattr(distributed, "state_dict"):
            return distributed
        state_dict_owner = distributed
        state_dict_name = "state_dict"
        had_own_state_dict = state_dict_name in state_dict_owner.__dict__
        original_state_dict = state_dict_owner.state_dict
        original_state_dict_descriptor = (
            inspect.getattr_static(state_dict_owner, state_dict_name)
            if had_own_state_dict
            else None
        )

        def state_dict(*state_dict_args, **state_dict_kwargs):
            values = original_state_dict(*state_dict_args, **state_dict_kwargs)
            for name in native_names:
                parameter = values.get(name)
                if parameter is not None:
                    native_dtensor_parameters[id(parameter)] = parameter
                    native_dtensor_names[id(parameter)] = name
            return values

        setattr(state_dict_owner, state_dict_name, state_dict)
        state_dict_restorations.append(
            (state_dict_owner, had_own_state_dict, original_state_dict_descriptor)
        )
        return distributed

    def dtensor_init(operation, parameter):
        original_dtensor_init(operation, parameter)
        operation._axolotl_native_nvfp4_component = (
            native_dtensor_parameters.get(id(parameter)) is parameter
        )
        operation._axolotl_native_nvfp4_parameter = parameter

    def shard_tensor(operation, tensor, *args, **kwargs):
        if getattr(operation, "_axolotl_native_nvfp4_component", False):
            device = kwargs.get("device")
            operation._axolotl_native_nvfp4_device = device
            parameter = getattr(operation, "_axolotl_native_nvfp4_parameter", None)
            if parameter is not None:
                native_dtensor_devices[id(parameter)] = device
                name = native_dtensor_names.get(id(parameter))
                if name is not None:
                    native_dtensor_devices_by_name[name] = device
            return tensor[...]
        return original_shard_tensor(operation, tensor, *args, **kwargs)

    def preprocess(quantizer, model, **kwargs):
        original_preprocess(quantizer, model, **kwargs)
        if (
            type(quantizer.quantization_config.quant_type).__name__
            != "NVFP4WeightOnlyConfig"
        ):
            return
        mesh = _tp_mesh(device_mesh)
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        native_names = set()
        for name, payload in getattr(quantizer, "metadata", {}).items():
            try:
                metadata = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                continue
            if _metadata_has_nvfp4(metadata):
                native_names.add(name)
        grouped = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            grouped.setdefault(id(parameter), []).append((name, parameter))
        manifest = {}
        for aliases in grouped.values():
            native_aliases = [name for name, _ in aliases if name in native_names]
            if not native_aliases:
                continue
            plans = [_native_nvfp4_tp_plan(model, name) for name, _ in aliases]
            if any(plan is None for plan in plans) and any(
                plan is not None for plan in plans
            ):
                raise ValueError(
                    f"Native NVFP4 TP aliases mix replicated and sharded placement: {native_aliases[0]}"
                )
            placements = []
            for name, parameter in aliases:
                if name not in native_names:
                    continue
                plan = _native_nvfp4_tp_plan(model, name)
                if plan is None:
                    continue
                shard = _native_nvfp4_tp_shard_shape(
                    parameter.shape,
                    _DENSE_TP_DIMS[plan],
                    rank,
                    world_size,
                    block_size=16,
                    name=name,
                )
                placements.append(shard)
                manifest[name] = shard
            if placements and any(
                (shard.dim, shard.start, shard.end)
                != (placements[0].dim, placements[0].start, placements[0].end)
                for shard in placements[1:]
            ):
                raise ValueError(
                    f"Native NVFP4 TP aliases disagree on shard placement: {native_aliases[0]}"
                )
        model._axolotl_native_nvfp4_tp_manifest = manifest

    def get_weight_conversions(quantizer):
        converters = original(quantizer)
        if (
            type(quantizer.quantization_config.quant_type).__name__
            != "NVFP4WeightOnlyConfig"
        ):
            return converters
        return [
            _native_nvfp4_weight_converter(
                converter,
                device_mesh,
                native_dtensor_devices,
                native_dtensor_devices_by_name,
            )
            for converter in converters
        ]

    try:
        TorchAoHfQuantizer.get_weight_conversions = get_weight_conversions
        TorchAoHfQuantizer._process_model_before_weight_loading = preprocess
        distribute_owner.maybe_distribute_model = classmethod(distribute_model)
        DtensorShardOperation.__init__ = dtensor_init
        DtensorShardOperation.shard_tensor = shard_tensor
        yield
    finally:
        TorchAoHfQuantizer.get_weight_conversions = original
        TorchAoHfQuantizer._process_model_before_weight_loading = original_preprocess
        if had_own_distribute_model:
            setattr(distribute_owner, distribute_name, original_distribute_descriptor)
        else:
            delattr(distribute_owner, distribute_name)
        for owner, had_own_state_dict, descriptor in reversed(state_dict_restorations):
            if had_own_state_dict:
                owner.state_dict = descriptor
            else:
                delattr(owner, "state_dict")
        native_dtensor_parameters.clear()
        DtensorShardOperation.__init__ = original_dtensor_init
        DtensorShardOperation.shard_tensor = original_shard_tensor
