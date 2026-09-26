"""Merge-aware PEFT LoRA support for native NVFP4 tensor parallelism."""

from __future__ import annotations

import hashlib
import types
import weakref

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, process_group):
        if dist.get_world_size(process_group) > 1:
            dist.all_reduce(value, group=process_group)
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, process_group):
        ctx.process_group = process_group
        return value

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        if dist.get_world_size(ctx.process_group) > 1:
            dist.all_reduce(grad, group=ctx.process_group)
        return grad, None


def _tp_group(mesh):
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    return (
        mesh.get_group("tp")
        if mesh_dim_names and "tp" in mesh_dim_names
        else mesh.get_group()
    )


def _is_native_nvfp4_dtensor(weight) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return (
        isinstance(weight, DTensor)
        and type(weight.to_local()).__name__ == "NVFP4Tensor"
    )


def _lora_tp_plan(model, module_name):
    from axolotl.monkeypatch.torchao_tp import _native_nvfp4_tp_plan

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    marker = "model.layers."
    position = module_name.find(marker)
    layer_name = module_name[position:] if position >= 0 else module_name
    return _native_nvfp4_tp_plan(base_model, layer_name)


def _tp_lora_layout(weight, plan) -> str:
    mesh = weight.device_mesh
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if mesh_dim_names and "tp" in mesh_dim_names:
        tp_axis = mesh_dim_names.index("tp")
    elif getattr(mesh, "ndim", 1) == 1:
        tp_axis = 0
    else:
        raise ValueError("Native NVFP4 TP LoRA requires a named TP mesh axis")
    if len(weight.placements) != getattr(mesh, "ndim", len(weight.placements)):
        raise ValueError("Native NVFP4 TP LoRA weight placements do not match its mesh")
    for axis, placement in enumerate(weight.placements):
        if axis == tp_axis:
            if type(placement).__name__ != "Shard" or placement.dim % 2 not in (0, 1):
                raise ValueError(
                    "Native NVFP4 TP LoRA requires an output- or input-axis TP shard"
                )
        elif type(placement).__name__ != "Replicate":
            raise ValueError(
                "Native NVFP4 TP LoRA only supports replicated non-TP mesh axes"
            )
    layout = "colwise" if weight.placements[tp_axis].dim % 2 == 0 else "rowwise"
    if plan is not None and plan != layout:
        raise ValueError(
            "Native NVFP4 TP LoRA weight placement disagrees with its TP plan"
        )
    return layout


def _merge_aware_unsupported_reason(module) -> str | None:
    local_weight = module.base_layer.weight.to_local()
    if getattr(local_weight, "act_quant_kwargs", None) is not None:
        return "dynamic NVFP4 activation quantization"
    if (
        getattr(local_weight, "per_tensor_scale", None) is not None
        and local_weight.per_tensor_scale.numel() != 1
    ):
        return "non-scalar NVFP4 per-tensor scales"
    return None


def _validate_ordinary_lora(module, layout: str) -> None:
    if getattr(module, "lora_variant", None) or any(
        getattr(module, "use_dora", {}).values()
    ):
        raise ValueError("Native NVFP4 TP supports ordinary LoRA adapters only")
    base_weight = module.base_layer.weight
    local_weight = base_weight.to_local()
    for adapter, lora_a in module.lora_A.items():
        lora_b = module.lora_B[adapter]
        if (
            type(lora_a.weight).__name__ == "DTensor"
            or type(lora_b.weight).__name__ == "DTensor"
        ):
            raise ValueError(
                "Native NVFP4 TP LoRA factors must be plain local parameters"
            )
        if lora_a.bias is not None or lora_b.bias is not None:
            raise ValueError("Native NVFP4 TP LoRA supports biasless factors only")
        if getattr(module.lora_dropout[adapter], "p", 0.0) != 0.0:
            raise ValueError("Native NVFP4 TP LoRA requires lora_dropout: 0")
        valid = lora_a.weight.shape[1] == (
            base_weight.shape[1] if layout == "colwise" else local_weight.shape[1]
        ) and lora_b.weight.shape[0] == (
            local_weight.shape[0] if layout == "colwise" else base_weight.shape[0]
        )
        if not valid:
            raise ValueError(
                f"Native NVFP4 TP {layout} LoRA factors do not match the base shard for adapter {adapter!r}"
            )


def _sum_gradient(process_group):
    def hook(grad):
        if dist.get_world_size(process_group) > 1:
            dist.all_reduce(grad, group=process_group)
        return grad

    return hook


def _warn_fallback(module, reason: str) -> None:
    if not getattr(module, "_axolotl_native_nvfp4_tp_merge_warning", False):
        LOG.warning(
            "Native NVFP4 TP LoRA merge-aware forward is unavailable (%s); using TP-correct ordinary LoRA, so merged NVFP4 deployment parity is not guaranteed.",
            reason,
        )
        module._axolotl_native_nvfp4_tp_merge_warning = True
    module._axolotl_merge_aware_unsupported = True


def _merge_aware_tp_forward(module, layout: str, process_group):
    original_forward = module.forward
    ordinary_forward = _ordinary_lora_tp_forward(module, layout, process_group)

    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {key: kwargs.pop(key, None) for key in VARIANT_KWARG_KEYS}
        if self.disable_adapters or self.merged:
            return original_forward(
                x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs
            )
        adapters = [
            adapter for adapter in self.active_adapters if adapter in self.lora_A
        ]
        if adapter_names is not None:
            raise ValueError(
                "Native NVFP4 TP LoRA does not support mixed-batch adapters"
            )
        if len(adapters) != 1:
            _warn_fallback(self, "multiple active adapters")
            return ordinary_forward(x, *args, **variant_kwargs, **kwargs)

        adapter = adapters[0]
        lora_a = self.lora_A[adapter].weight
        lora_b = self.lora_B[adapter].weight
        base = self.base_layer
        local_base = base.weight.to_local()
        lora_input = self._cast_input_dtype(x, lora_a.dtype)
        effective = (
            local_base.dequantize().float()
            + (lora_b.float() @ lora_a.float()) * self.scaling[adapter]
        ).to(lora_a.dtype)
        from axolotl.monkeypatch.torchao_nvfp4_merge import (
            quantize_native_effective_weight,
        )

        snapped_native = quantize_native_effective_weight(
            local_base, lora_a, lora_b, self.scaling[adapter]
        )
        snapped = snapped_native.dequantize(effective.dtype)
        weight = effective + (snapped - effective).detach()
        bias = getattr(base, "bias", None)
        if type(bias).__name__ == "DTensor":
            bias = bias.to_local()
        if layout == "colwise":
            if lora_input.requires_grad:
                lora_input = _IdentityForwardAllReduceBackward.apply(
                    lora_input, process_group
                )
            result = F.linear(lora_input, weight, bias)
        else:
            result = F.linear(lora_input, weight)
            result = _AllReduceForwardIdentityBackward.apply(result, process_group)
            if bias is not None:
                result = result + bias
        return result.to(x.dtype)

    return types.MethodType(forward, module)


def _ordinary_lora_tp_forward(module, layout: str, process_group):
    original_forward = module.forward

    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {key: kwargs.pop(key, None) for key in VARIANT_KWARG_KEYS}
        if self.disable_adapters or self.merged:
            return original_forward(
                x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs
            )
        if adapter_names is not None:
            raise ValueError(
                "Native NVFP4 TP LoRA does not support mixed-batch adapters"
            )
        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype
        for adapter in self.active_adapters:
            if adapter not in self.lora_A:
                continue
            lora_a = self.lora_A[adapter]
            lora_input = self._cast_input_dtype(x, lora_a.weight.dtype)
            if layout == "colwise" and lora_input.requires_grad:
                lora_input = _IdentityForwardAllReduceBackward.apply(
                    lora_input, process_group
                )
            delta = (
                self.lora_B[adapter](lora_a(self.lora_dropout[adapter](lora_input)))
                * self.scaling[adapter]
            )
            if layout == "rowwise":
                delta = _AllReduceForwardIdentityBackward.apply(delta, process_group)
            result = result + delta
        return result.to(result_dtype)

    return types.MethodType(forward, module)


def prepare_native_nvfp4_tp_lora(model, *, merge_aware: bool = True) -> bool:
    """Install TP-correct ordinary or static merge-aware native NVFP4 LoRA."""
    candidates = []
    for module_name, module in model.named_modules():
        base = getattr(module, "base_layer", None)
        weight = getattr(base, "weight", None)
        if not _is_native_nvfp4_dtensor(weight) or not hasattr(module, "lora_A"):
            continue
        plan = _lora_tp_plan(model, module_name)
        if plan not in ("colwise", "rowwise"):
            raise ValueError(
                "Native NVFP4 TP LoRA only supports colwise and rowwise TP plans"
            )
        layout = _tp_lora_layout(weight, plan)
        _validate_ordinary_lora(module, layout)
        unavailable = _merge_aware_unsupported_reason(module) if merge_aware else None
        if unavailable is not None:
            _warn_fallback(module, unavailable)
        elif not merge_aware:
            _warn_fallback(module, "merge-aware forward disabled by configuration")
        candidates.append(
            (
                module,
                layout,
                _tp_group(weight.device_mesh),
                unavailable is None and merge_aware,
            )
        )

    for module, layout, process_group, use_merge_aware in candidates:
        replicated_factors = module.lora_A if layout == "colwise" else module.lora_B
        hooked = getattr(module, "_axolotl_native_nvfp4_tp_lora_hooked_factors", None)
        if not isinstance(hooked, weakref.WeakValueDictionary):
            hooked = weakref.WeakValueDictionary()
        for factor in replicated_factors.values():
            if id(factor.weight) not in hooked:
                factor.weight.register_hook(_sum_gradient(process_group))
                hooked[id(factor.weight)] = factor.weight
        module._axolotl_native_nvfp4_tp_lora_hooked_factors = hooked
        if not getattr(module, "_axolotl_native_nvfp4_tp_lora_prepared", False):
            module.forward = (
                _merge_aware_tp_forward(module, layout, process_group)
                if use_merge_aware
                else _ordinary_lora_tp_forward(module, layout, process_group)
            )
            module._axolotl_native_nvfp4_tp_lora_prepared = True
    return bool(candidates)


def _all_gather_factor(
    value: torch.Tensor, process_group, *, shard_dim: int | None = None
) -> list[torch.Tensor]:
    value = value.detach().contiguous()
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return [value]
    if not dist.is_initialized():
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(gathered, value, group=process_group)
        return gathered
    shape = torch.tensor(value.shape, device=value.device, dtype=torch.int64)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=process_group)
    shape_values = [tuple(item.tolist()) for item in shapes]
    if all(item == shape_values[0] for item in shape_values[1:]):
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(gathered, value, group=process_group)
        return gathered
    if shard_dim is None or any(
        any(
            actual[axis] != value.shape[axis]
            for axis in range(value.ndim)
            if axis != shard_dim
        )
        for actual in shape_values
    ):
        raise ValueError("Native NVFP4 TP LoRA factor shapes disagree across TP ranks")
    padded_shape = list(value.shape)
    padded_shape[shard_dim] = max(item[shard_dim] for item in shape_values)
    padded = value.new_zeros(padded_shape)
    padded.narrow(shard_dim, 0, value.shape[shard_dim]).copy_(value)
    gathered_padded = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_padded, padded, group=process_group)
    return [
        item.narrow(shard_dim, 0, actual[shard_dim]).contiguous()
        for item, actual in zip(gathered_padded, shape_values, strict=True)
    ]


def _sync_export_failure(error: str | None) -> str | None:
    if not dist.is_initialized():
        return error
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    return next((item for item in errors if item is not None), None)


def _replicated_factor_for_save(value: torch.Tensor, process_group, name: str):
    replicas = _all_gather_factor(value, process_group)
    if any(not torch.equal(replica, replicas[0]) for replica in replicas[1:]):
        raise ValueError(
            f"Native NVFP4 TP LoRA replicated factor {name!r} differs across TP ranks"
        )
    return replicas[0]


def _tp_export_modules(model):
    modules = []
    for module_name, module in model.named_modules():
        if not getattr(module, "_axolotl_native_nvfp4_tp_lora_prepared", False):
            continue
        weight = module.base_layer.weight
        if not _is_native_nvfp4_dtensor(weight):
            continue
        layout = _tp_lora_layout(weight, _lora_tp_plan(model, module_name))
        adapters = tuple(sorted(module.lora_A))
        modules.append(
            (module_name, module, layout, _tp_group(weight.device_mesh), adapters)
        )
    return modules


def _tp_export_preflight(modules) -> None:
    specification = tuple(
        (
            name,
            layout,
            adapters,
        )
        for name, module, layout, _group, adapters in modules
    )
    if not dist.is_initialized():
        return
    specifications = [None] * dist.get_world_size()
    dist.all_gather_object(specifications, specification)
    if any(item != specifications[0] for item in specifications[1:]):
        raise ValueError("Native NVFP4 TP LoRA export layout differs across ranks")


def _update_export_digest(digest, name: str, value: torch.Tensor) -> None:
    digest.update(name.encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def native_nvfp4_tp_peft_state_dict(model, *, collect_on_this_rank: bool):
    """Gather TP-local ordinary LoRA factors for PEFT's normal adapter save path."""
    try:
        modules = _tp_export_modules(model)
    except Exception as exc:  # pylint: disable=broad-except
        failure = _sync_export_failure(f"{type(exc).__name__}: {exc}")
        if failure is not None:
            raise RuntimeError(
                f"Native NVFP4 TP adapter export failed: {failure}"
            ) from exc
        raise
    if not modules:
        return None
    try:
        _tp_export_preflight(modules)
    except Exception as exc:  # pylint: disable=broad-except
        failure = _sync_export_failure(f"{type(exc).__name__}: {exc}")
        if failure is not None:
            raise RuntimeError(
                f"Native NVFP4 TP adapter export failed: {failure}"
            ) from exc
        raise

    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    names_by_value: dict[int, list[str]] = {}
    for name, value in named_parameters.items():
        names_by_value.setdefault(id(value), []).append(name)
    full_state = (
        {
            name: value.detach().cpu().clone()
            for name, value in named_parameters.items()
            if "lora_" in name
        }
        if collect_on_this_rank
        else {}
    )
    digest = hashlib.sha256()
    failure = None
    for module_name, module, layout, process_group, adapters in modules:
        is_tp_leader = not dist.is_initialized() or dist.get_rank(process_group) == 0
        for adapter in adapters:
            lora_a = module.lora_A[adapter].weight
            lora_b = module.lora_B[adapter].weight
            local, replicated, shard_dim = (
                (lora_b, lora_a, 0) if layout == "colwise" else (lora_a, lora_b, 1)
            )
            full_local = torch.cat(
                _all_gather_factor(local, process_group, shard_dim=shard_dim),
                dim=shard_dim,
            )
            try:
                full_replicated = _replicated_factor_for_save(
                    replicated, process_group, f"{module_name}.{adapter}"
                )
                for value, gathered in (
                    (local, full_local),
                    (replicated, full_replicated),
                ):
                    names = names_by_value.get(id(value))
                    if not names:
                        raise ValueError(
                            "Native NVFP4 TP LoRA factor is missing from named parameters"
                        )
                    if is_tp_leader:
                        for name in names:
                            _update_export_digest(digest, name, gathered)
                    if collect_on_this_rank:
                        for name in names:
                            full_state[name] = gathered.detach().cpu().clone()
            except Exception as exc:  # pylint: disable=broad-except
                failure = f"{type(exc).__name__}: {exc}"
    failure = _sync_export_failure(failure)
    if failure is not None:
        raise RuntimeError(f"Native NVFP4 TP adapter export failed: {failure}")

    if dist.is_initialized():
        digests = [None] * dist.get_world_size()
        local_digest = digest.hexdigest() if is_tp_leader else None
        dist.all_gather_object(digests, local_digest)
        leaders = [item for item in digests if item is not None]
        if any(item != leaders[0] for item in leaders[1:]):
            raise RuntimeError(
                "Native NVFP4 TP adapter export failed: factors differ across TP groups"
            )
    return full_state if collect_on_this_rank else {}
