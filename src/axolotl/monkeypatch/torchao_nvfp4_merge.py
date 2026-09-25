"""Merge-aware LoRA forward helpers for native TorchAO NVFP4 weights."""

from __future__ import annotations

import base64
import hashlib
import json
import types
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _tensor_snapshot(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    raw = (
        value.detach()
        .reshape(-1)
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes": base64.b64encode(raw).decode("ascii"),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if hasattr(value, "__dict__"):
        return {
            key: _json_value(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return repr(value)


@dataclass(frozen=True)
class NativeNVFP4Recipe:
    """The complete native NVFP4 encoding contract for one frozen weight."""

    block_size: int
    orig_dtype: torch.dtype
    per_tensor_scale: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: Any

    def quantize(self, weight: torch.Tensor):
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor.to_nvfp4(
            weight.to(self.orig_dtype),
            block_size=self.block_size,
            per_tensor_scale=self.per_tensor_scale,
            act_per_tensor_scale=self.act_per_tensor_scale,
            is_swizzled_scales=self.is_swizzled_scales,
            use_triton_kernel=self.use_triton_kernel,
            act_quant_kwargs=self.act_quant_kwargs,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe encoding identity for adapter export validation."""
        return {
            "block_size": self.block_size,
            "orig_dtype": str(self.orig_dtype),
            "per_tensor_scale": _tensor_snapshot(self.per_tensor_scale),
            "act_per_tensor_scale": _tensor_snapshot(self.act_per_tensor_scale),
            "is_swizzled_scales": self.is_swizzled_scales,
            "use_triton_kernel": self.use_triton_kernel,
            "act_quant_kwargs": _json_value(self.act_quant_kwargs),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def capture_native_nvfp4_recipe(weight) -> NativeNVFP4Recipe:
    """Capture the encoding contract from an actual TorchAO ``NVFP4Tensor``."""
    if type(weight).__name__ != "NVFP4Tensor":
        raise TypeError(f"expected TorchAO NVFP4Tensor, got {type(weight).__name__}")
    return NativeNVFP4Recipe(
        block_size=weight.block_size,
        orig_dtype=weight.orig_dtype,
        per_tensor_scale=weight.per_tensor_scale,
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=weight.is_swizzled_scales,
        use_triton_kernel=weight.use_triton_kernel,
        act_quant_kwargs=weight.act_quant_kwargs,
    )


def quantize_native_effective_weight(base_weight, lora_a, lora_b, scaling):
    """Encode ``dequant(base) + scaling * (B @ A)`` with the base's native recipe."""
    recipe = capture_native_nvfp4_recipe(base_weight)
    with torch.autocast(device_type=lora_a.device.type, enabled=False):
        delta = (lora_b @ lora_a) * scaling
        effective = (base_weight.dequantize().float() + delta.float()).to(
            recipe.orig_dtype
        )
    return recipe.quantize(effective)


def _native_merge_aware_forward(
    ctx, inputs, bias, base_weight, lora_a, lora_b, scaling
):
    snapped = quantize_native_effective_weight(base_weight, lora_a, lora_b, scaling)
    output = F.linear(inputs, snapped, bias)
    ctx.save_for_backward(inputs, lora_a, lora_b, snapped.dequantize())
    ctx.scaling = scaling
    ctx.bias_requires_grad = bias is not None and bias.requires_grad
    return output


def _native_merge_aware_backward(ctx, grad_output):
    inputs, lora_a, lora_b, snapped = ctx.saved_tensors
    grad_inputs = grad_output.matmul(snapped)
    flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    grad_effective = flat_grad.transpose(0, 1).matmul(flat_inputs)
    grad_b = (
        grad_effective.to(lora_b.dtype).matmul(lora_a.to(lora_b.dtype).transpose(0, 1))
        * ctx.scaling
    )
    grad_a = (
        lora_b.to(lora_a.dtype).transpose(0, 1).matmul(grad_effective.to(lora_a.dtype))
        * ctx.scaling
    )
    grad_bias = None
    if ctx.bias_requires_grad:
        grad_bias = flat_grad.sum(dim=0)
    return grad_inputs, grad_bias, None, grad_a, grad_b, None


class _NativeNVFP4MergeAwareLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bias, base_weight, lora_a, lora_b, scaling):
        return _native_merge_aware_forward(
            ctx, inputs, bias, base_weight, lora_a, lora_b, scaling
        )

    @staticmethod
    def backward(ctx, grad_output):
        return _native_merge_aware_backward(ctx, grad_output)


def native_nvfp4_merge_aware_linear(inputs, bias, base_weight, lora_a, lora_b, scaling):
    """Apply the native snapped effective weight with straight-through LoRA gradients."""
    if getattr(base_weight, "act_quant_kwargs", None) is not None:
        raise ValueError(
            "native merge-aware linear requires static activation precision"
        )
    return _NativeNVFP4MergeAwareLinear.apply(
        inputs, bias, base_weight, lora_a, lora_b, scaling
    )


def _unsupported_native_merge_aware(module, reason: str) -> None:
    if getattr(module, "_axolotl_merge_aware_unsupported", False):
        return
    module._axolotl_merge_aware_unsupported = True
    owner_ref = getattr(module, "_axolotl_native_nvfp4_owner", None)
    owner = owner_ref() if isinstance(owner_ref, weakref.ReferenceType) else None
    if owner is not None:
        owner._axolotl_merge_aware_unsupported = True
    LOG.warning(
        "NVFP4 MERGE WARNING: native merge-aware LoRA unavailable for %s (%s); "
        "merged NVFP4 deployment parity is not guaranteed",
        getattr(module, "_axolotl_native_nvfp4_name", type(module).__name__),
        reason,
    )


def _native_merge_aware_reason(module, adapters) -> str | None:
    if len(adapters) > 1:
        return "multiple active adapters"
    if any(
        getattr(dropout, "p", 0.0) != 0.0 for dropout in module.lora_dropout.values()
    ):
        return "LoRA dropout"
    if any(module.lora_bias.values()):
        return "LoRA bias"
    if module.lora_variant:
        return "LoRA variant"
    if any(module.use_dora.get(adapter) for adapter in module.lora_A):
        return "DoRA"
    return None


def _native_nvfp4_lora_forward(self, x, *args, **kwargs):
    if self.disable_adapters or self.merged:
        return self._axolotl_native_nvfp4_orig_forward(x, *args, **kwargs)
    if kwargs.get("adapter_names") is not None:
        _unsupported_native_merge_aware(self, "per-sample adapter_names")
        return self._axolotl_native_nvfp4_orig_forward(x, *args, **kwargs)
    adapters = [adapter for adapter in self.active_adapters if adapter in self.lora_A]
    if not adapters:
        return self._axolotl_native_nvfp4_orig_forward(x, *args, **kwargs)
    if reason := _native_merge_aware_reason(self, adapters):
        _unsupported_native_merge_aware(self, reason)
        return self._axolotl_native_nvfp4_orig_forward(x, *args, **kwargs)
    adapter = adapters[0]
    base = self.get_base_layer()
    return native_nvfp4_merge_aware_linear(
        x,
        base.bias,
        base.weight,
        self.lora_A[adapter].weight,
        self.lora_B[adapter].weight,
        self.scaling[adapter],
    )


def install_native_nvfp4_merge_aware_lora_linears(model: torch.nn.Module) -> int:
    """Install native NVFP4 merge-aware PEFT forwards, warning on unsupported wrappers."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    installed = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        base = module.get_base_layer()
        if type(getattr(base, "weight", None)).__name__ != "NVFP4Tensor":
            continue
        if hasattr(module, "_axolotl_native_nvfp4_orig_forward"):
            installed += 1
            continue
        adapters = [
            adapter for adapter in module.active_adapters if adapter in module.lora_A
        ]
        reason = _native_merge_aware_reason(module, adapters)
        if getattr(base.weight, "ndim", 0) != 2:
            reason = "non-matrix base weight"
        elif getattr(base.weight, "act_quant_kwargs", None) is not None:
            reason = "dynamic activation quantization"
        module._axolotl_native_nvfp4_owner = weakref.ref(model)
        module._axolotl_native_nvfp4_name = name
        if reason:
            _unsupported_native_merge_aware(module, reason)
            continue
        module._axolotl_native_nvfp4_orig_forward = module.forward
        module._ma_orig_forward = module.forward
        module.forward = types.MethodType(_native_nvfp4_lora_forward, module)
        installed += 1
    return installed
