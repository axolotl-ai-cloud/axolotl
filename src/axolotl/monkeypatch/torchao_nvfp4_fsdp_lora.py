"""FSDP2-safe native-NVFP4 LoRA merge-aware forwards."""

from __future__ import annotations

import types
import weakref

import torch

from axolotl.monkeypatch.torchao_nvfp4_merge import (
    _native_merge_aware_reason,
    _unsupported_native_merge_aware,
    native_nvfp4_merge_aware_linear,
)


def _native_weight(weight):
    return getattr(weight, "_local_tensor", weight)


def _install_materialize_mode(module):
    if hasattr(module, "_axolotl_materialize_orig_forward"):
        return
    original = module.forward

    def forward(self, *args, **kwargs):
        if kwargs.pop("_axolotl_materialize_weight", False):
            return self.weight.clone()
        return self._axolotl_materialize_orig_forward(*args, **kwargs)

    module._axolotl_materialize_orig_forward = original
    module.forward = types.MethodType(forward, module)


def _materialize_weight(module):
    return _native_weight(module(_axolotl_materialize_weight=True))


def _fsdp_native_forward(self, x, *args, **kwargs):
    if self.disable_adapters or self.merged:
        return self._axolotl_fsdp_native_orig_forward(x, *args, **kwargs)
    if kwargs.get("adapter_names") is not None:
        _unsupported_native_merge_aware(self, "per-sample adapter_names")
        return self._axolotl_fsdp_native_orig_forward(x, *args, **kwargs)
    adapters = [adapter for adapter in self.active_adapters if adapter in self.lora_A]
    if not adapters:
        return self._axolotl_fsdp_native_orig_forward(x, *args, **kwargs)
    if reason := _native_merge_aware_reason(self, adapters):
        _unsupported_native_merge_aware(self, reason)
        return self._axolotl_fsdp_native_orig_forward(x, *args, **kwargs)
    base = self.get_base_layer()
    base_weight = _materialize_weight(base)
    if (
        len(adapters) != 1
        or base.bias is not None
        or getattr(base_weight, "ndim", 0) != 2
    ):
        return self._axolotl_fsdp_native_orig_forward(x, *args, **kwargs)
    adapter = adapters[0]
    return native_nvfp4_merge_aware_linear(
        x,
        None,
        base_weight,
        _materialize_weight(self.lora_A[adapter]),
        _materialize_weight(self.lora_B[adapter]),
        self.scaling[adapter],
    )


def install_fsdp_native_nvfp4_merge_aware_lora_linears(model: torch.nn.Module) -> int:
    """Install native NVFP4 forwards after FSDP2 wraps all children."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    installed = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        base = module.get_base_layer()
        native_weight = _native_weight(getattr(base, "weight", None))
        if type(native_weight).__name__ != "NVFP4Tensor":
            continue
        if hasattr(module, "_axolotl_fsdp_native_orig_forward"):
            installed += 1
            continue
        adapters = [
            adapter for adapter in module.active_adapters if adapter in module.lora_A
        ]
        if not adapters:
            continue
        reason = _native_merge_aware_reason(module, adapters)
        if len(adapters) != 1:
            reason = reason or "multiple active adapters"
        elif base.bias is not None:
            reason = "base bias"
        elif getattr(native_weight, "ndim", 0) != 2:
            reason = "non-matrix base weight"
        module._axolotl_native_nvfp4_owner = weakref.ref(model)
        module._axolotl_native_nvfp4_name = name
        if reason:
            _unsupported_native_merge_aware(module, reason)
            continue
        adapter = adapters[0]
        _install_materialize_mode(base)
        _install_materialize_mode(module.lora_A[adapter])
        _install_materialize_mode(module.lora_B[adapter])
        module._axolotl_fsdp_native_orig_forward = module.forward
        module._ma_orig_forward = module.forward
        module.forward = types.MethodType(_fsdp_native_forward, module)
        installed += 1
    return installed
