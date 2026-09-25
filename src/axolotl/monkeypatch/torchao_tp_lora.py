"""Ordinary PEFT LoRA support for native NVFP4 tensor parallelism."""

from __future__ import annotations

import types
import weakref

import torch
import torch.distributed as dist
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    """Sum a TP partial in forward without changing its local backward."""

    @staticmethod
    def forward(ctx, value, process_group):
        if dist.get_world_size(process_group) > 1:
            dist.all_reduce(value, group=process_group)
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    """Preserve a replicated value and sum its partial input gradient."""

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
        if layout == "colwise":
            valid = (
                lora_a.weight.shape[1] == base_weight.shape[1]
                and lora_b.weight.shape[0] == local_weight.shape[0]
            )
        else:
            valid = (
                lora_a.weight.shape[1] == local_weight.shape[1]
                and lora_b.weight.shape[0] == base_weight.shape[0]
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
                "Native NVFP4 TP does not support mixed-batch LoRA adapters"
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


def prepare_native_nvfp4_tp_lora(model) -> bool:
    """Install TP-correct ordinary LoRA math after PEFT wraps a native NVFP4 model."""
    candidates = []
    for module_name, module in model.named_modules():
        base = getattr(module, "base_layer", None)
        weight = getattr(base, "weight", None)
        if not _is_native_nvfp4_dtensor(weight):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        plan = _lora_tp_plan(model, module_name)
        if plan not in ("colwise", "rowwise"):
            raise ValueError(
                "Native NVFP4 TP LoRA only supports colwise and rowwise TP plans"
            )
        layout = _tp_lora_layout(weight, plan)
        _validate_ordinary_lora(module, layout)
        candidates.append((module, layout, _tp_group(weight.device_mesh)))

    for module, layout, process_group in candidates:
        replicated_factors = module.lora_A if layout == "colwise" else module.lora_B
        hooked_factors = getattr(
            module, "_axolotl_native_nvfp4_tp_lora_hooked_factors", None
        )
        if not isinstance(hooked_factors, weakref.WeakValueDictionary):
            hooked_factors = weakref.WeakValueDictionary()
        for factor in replicated_factors.values():
            factor_id = id(factor.weight)
            if factor_id not in hooked_factors:
                factor.weight.register_hook(_sum_gradient(process_group))
                hooked_factors[factor_id] = factor.weight
        module._axolotl_native_nvfp4_tp_lora_hooked_factors = hooked_factors
        if not getattr(module, "_axolotl_native_nvfp4_tp_lora_prepared", False):
            module.forward = _ordinary_lora_tp_forward(module, layout, process_group)
            module._axolotl_native_nvfp4_tp_lora_prepared = True
    return bool(candidates)
