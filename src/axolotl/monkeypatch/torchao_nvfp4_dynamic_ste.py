"""Input-gradient STEs for frozen dynamic native NVFP4 linears."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _native_weight(weight):
    return getattr(weight, "_local_tensor", weight)


def _is_frozen_dynamic_native_linear(module: torch.nn.Module) -> bool:
    weight = _native_weight(getattr(module, "weight", None))
    return (
        isinstance(module, torch.nn.Linear)
        and type(weight).__name__ == "NVFP4Tensor"
        and getattr(weight, "act_quant_kwargs", None) is not None
        and not weight.requires_grad
    )


class _FrozenDynamicNVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bias, weight):
        ctx.input_dtype = inputs.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.save_for_backward(weight.clone())
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            dense_weight = weight.dequantize()
            grad_inputs = grad_output.to(dense_weight.dtype).matmul(dense_weight)
            grad_inputs = grad_inputs.to(ctx.input_dtype)
        grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
            grad_bias = grad_bias.to(ctx.bias_dtype)
        return grad_inputs, grad_bias, None


def native_nvfp4_frozen_dynamic_linear(inputs, bias, weight):
    """Run native dynamic NVFP4 forward with a frozen-base input-gradient STE."""
    return _FrozenDynamicNVFP4Linear.apply(inputs, bias, weight)


def _full_native_weight(module: torch.nn.Module):
    weight = getattr(module, "weight", None)
    if type(weight).__name__ != "NVFP4Tensor":
        raise RuntimeError(
            "dynamic NVFP4 STE requires the owning forward to receive a full "
            "native NVFP4Tensor"
        )
    return weight


def _install_dynamic_forward(module, weight_getter):
    if hasattr(module, "_axolotl_dynamic_nvfp4_ste_orig_forward"):
        return False
    original = module.forward

    def forward(self, x=None, *args, **kwargs):
        if kwargs.pop("_axolotl_materialize_weight", False):
            return weight_getter(self).clone()
        if x is None:
            raise TypeError("dynamic NVFP4 linear forward requires an input tensor")
        return native_nvfp4_frozen_dynamic_linear(x, self.bias, weight_getter(self))

    module._axolotl_dynamic_nvfp4_ste_orig_forward = original
    module.forward = types.MethodType(forward, module)
    return True


def _is_dynamic_native_weight(weight) -> bool:
    weight = _native_weight(weight)
    return (
        type(weight).__name__ == "NVFP4Tensor"
        and getattr(weight, "act_quant_kwargs", None) is not None
    )


def _dynamic_native_parameters(model: torch.nn.Module):
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if _is_dynamic_native_weight(parameter):
            yield name, parameter


def _owner_module(model: torch.nn.Module, name):
    module_name, separator, _ = name.rpartition(".")
    return model.get_submodule(module_name) if separator else model


def _eligible_dynamic_native_linear(model: torch.nn.Module, name, parameter) -> bool:
    _, _, parameter_name = name.rpartition(".")
    if parameter_name != "weight":
        return False
    module = _owner_module(model, name)
    weight = _native_weight(parameter)
    return (
        isinstance(module, torch.nn.Linear)
        and not parameter.requires_grad
        and getattr(weight, "ndim", 0) == 2
    )


def dynamic_native_nvfp4_input_ste_targets(model: torch.nn.Module) -> tuple[str, ...]:
    """Return frozen 2-D dynamic native linear weights requiring input STEs."""
    return tuple(
        name
        for name, parameter in _dynamic_native_parameters(model)
        if _eligible_dynamic_native_linear(model, name, parameter)
    )


def _dynamic_native_linears(model: torch.nn.Module):
    return tuple(
        _owner_module(model, name)
        for name in dynamic_native_nvfp4_input_ste_targets(model)
    )


def _packed_dynamic_zero3_modules(model: torch.nn.Module):
    return tuple(
        module
        for module in model.modules()
        if getattr(module, "_axolotl_nvfp4_act_quant_kwargs", None) is not None
    )


def _zero3_native_forward_in_chain(module: torch.nn.Module) -> bool:
    forwards = (
        getattr(module, "forward", None),
        getattr(module, "_axolotl_dynamic_nvfp4_ste_orig_forward", None),
        getattr(module, "_axolotl_deepspeed_materialize_orig_forward", None),
    )
    return any(
        getattr(getattr(forward, "__func__", forward), "__name__", None)
        == "_zero3_native_forward"
        for forward in forwards
    )


def _valid_packed_dynamic_zero3(model: torch.nn.Module, module) -> bool:
    return bool(
        getattr(model, "_axolotl_native_nvfp4_zero3_components", ())
        and _zero3_native_forward_in_chain(module)
    )


def native_nvfp4_dynamic_input_ste_preflight(model: torch.nn.Module) -> bool:
    """Require every dynamic native parameter to have an eligible unpacked owner."""
    entries = tuple(_dynamic_native_parameters(model))
    return (
        bool(entries)
        and all(
            _eligible_dynamic_native_linear(model, name, parameter)
            for name, parameter in entries
        )
        and not _packed_dynamic_zero3_modules(model)
    )


def validate_native_nvfp4_dynamic_input_stes(model: torch.nn.Module) -> bool:
    """Mark a model only when every dynamic native base has a valid input-gradient path."""
    entries = tuple(_dynamic_native_parameters(model))
    missing = [
        name
        for name, parameter in entries
        if not _eligible_dynamic_native_linear(model, name, parameter)
        or not hasattr(
            _owner_module(model, name),
            "_axolotl_dynamic_nvfp4_ste_orig_forward",
        )
    ]
    packed = _packed_dynamic_zero3_modules(model)
    missing.extend(
        type(module).__name__
        for module in packed
        if not _valid_packed_dynamic_zero3(model, module)
    )
    valid = bool(entries or packed) and not missing
    model._axolotl_native_nvfp4_dynamic_input_gradients = valid
    return valid


def install_native_nvfp4_dynamic_input_stes(model: torch.nn.Module) -> int:
    """Install frozen dynamic-NVFP4 input STEs on unsharded native linears."""
    installed = 0
    for module in _dynamic_native_linears(model):
        installed += _install_dynamic_forward(module, lambda current: current.weight)
    validate_native_nvfp4_dynamic_input_stes(model)
    return installed


def install_fsdp_native_nvfp4_dynamic_input_stes(model: torch.nn.Module) -> int:
    """Install frozen dynamic input STEs after FSDP wraps every native linear."""
    installed = 0
    for module in _dynamic_native_linears(model):
        installed += _install_dynamic_forward(module, _full_native_weight)
    validate_native_nvfp4_dynamic_input_stes(model)
    return installed


def install_deepspeed_native_nvfp4_dynamic_input_stes(
    model: torch.nn.Module, *, weight_getter=_full_native_weight
) -> int:
    """Install frozen dynamic input STEs with a DeepSpeed owning-weight getter."""
    installed = 0
    for module in _dynamic_native_linears(model):
        installed += _install_dynamic_forward(module, weight_getter)
    validate_native_nvfp4_dynamic_input_stes(model)
    return installed
