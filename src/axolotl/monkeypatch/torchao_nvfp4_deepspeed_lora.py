"""DeepSpeed-safe static native-NVFP4 LoRA merge-aware forwards."""

from __future__ import annotations

import types
import weakref

import torch
from transformers import TrainerCallback

from axolotl.monkeypatch.torchao_nvfp4_merge import (
    _native_merge_aware_reason,
    _unsupported_native_merge_aware,
    native_nvfp4_merge_aware_linear,
)


def _is_zero3_native_base(module: torch.nn.Module) -> bool:
    return hasattr(module, "_axolotl_nvfp4_qdata")


def _native_base_is_frozen(module: torch.nn.Module) -> bool:
    if _is_zero3_native_base(module):
        return all(
            not getattr(module, name).requires_grad
            for name in ("_axolotl_nvfp4_qdata", "_axolotl_nvfp4_scale_bytes")
        )
    return not getattr(module.weight, "requires_grad", True)


def _native_recipe(module: torch.nn.Module):
    if _is_zero3_native_base(module):
        return (
            getattr(module.weight, "ndim", 0),
            getattr(module, "_axolotl_nvfp4_act_quant_kwargs", None),
        )
    weight = getattr(module, "weight", None)
    if type(weight).__name__ != "NVFP4Tensor":
        return None
    return weight.ndim, getattr(weight, "act_quant_kwargs", None)


def _install_materialize_mode(module: torch.nn.Module) -> None:
    if hasattr(module, "_axolotl_deepspeed_materialize_orig_forward"):
        return
    original = module.forward

    def forward(self, *args, **kwargs):
        if kwargs.pop("_axolotl_materialize_weight", False):
            if _is_zero3_native_base(self):
                from axolotl.monkeypatch.torchao_deepspeed import (
                    _zero3_reconstruct_native_weight,
                )

                return _zero3_reconstruct_native_weight(self).clone()
            return self.weight.clone()
        return self._axolotl_deepspeed_materialize_orig_forward(*args, **kwargs)

    module._axolotl_deepspeed_materialize_orig_forward = original
    module.forward = types.MethodType(forward, module)


def _materialize_weight(module: torch.nn.Module):
    return module(_axolotl_materialize_weight=True)


def _deepspeed_native_forward(self, x, *args, **kwargs):
    if self.disable_adapters:
        _unsupported_native_merge_aware(self, "disabled adapters")
        return self._axolotl_deepspeed_native_orig_forward(x, *args, **kwargs)
    if self.merged:
        _unsupported_native_merge_aware(self, "merged adapters")
        return self._axolotl_deepspeed_native_orig_forward(x, *args, **kwargs)
    if kwargs.get("adapter_names") is not None:
        _unsupported_native_merge_aware(self, "per-sample adapter_names")
        return self._axolotl_deepspeed_native_orig_forward(x, *args, **kwargs)
    adapters = [adapter for adapter in self.active_adapters if adapter in self.lora_A]
    if not adapters:
        _unsupported_native_merge_aware(self, "no active adapter")
        return self._axolotl_deepspeed_native_orig_forward(x, *args, **kwargs)
    if reason := _native_merge_aware_reason(self, adapters):
        _unsupported_native_merge_aware(self, reason)
        return self._axolotl_deepspeed_native_orig_forward(x, *args, **kwargs)
    base = self.get_base_layer()
    recipe = _native_recipe(base)
    if recipe is None:
        reason = "native base representation changed"
    elif len(adapters) != 1:
        reason = "multiple active adapters"
    elif base.bias is not None:
        reason = "base bias"
    elif recipe[0] != 2:
        reason = "non-matrix base weight"
    elif recipe[1] is not None:
        reason = "dynamic activation quantization"
    else:
        reason = None
    if reason:
        _unsupported_native_merge_aware(self, reason)
        return self._axolotl_deepspeed_native_orig_forward(x, *args, **kwargs)
    adapter = adapters[0]
    return native_nvfp4_merge_aware_linear(
        x,
        None,
        _materialize_weight(base),
        _materialize_weight(self.lora_A[adapter]),
        _materialize_weight(self.lora_B[adapter]),
        self.scaling[adapter],
    )


def install_deepspeed_native_nvfp4_merge_aware_lora_linears(
    model: torch.nn.Module,
) -> int:
    """Install static native-NVFP4 PEFT forwards after DeepSpeed wraps children."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    installed = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        base = module.get_base_layer()
        recipe = _native_recipe(base)
        if recipe is None:
            continue
        if hasattr(module, "_axolotl_deepspeed_native_orig_forward"):
            installed += 1
            continue
        adapters = [
            adapter for adapter in module.active_adapters if adapter in module.lora_A
        ]
        reason = _native_merge_aware_reason(module, adapters)
        if len(adapters) != 1:
            reason = reason or "multiple active adapters"
        elif base.bias is not None:
            reason = "base bias"
        elif not _native_base_is_frozen(base):
            reason = "trainable native base"
        elif recipe[0] != 2:
            reason = "non-matrix base weight"
        elif recipe[1] is not None:
            reason = "dynamic activation quantization"
        module._axolotl_native_nvfp4_owner = weakref.ref(model)
        module._axolotl_native_nvfp4_name = name
        if reason:
            _unsupported_native_merge_aware(module, reason)
            continue
        adapter = adapters[0]
        _install_materialize_mode(base)
        _install_materialize_mode(module.lora_A[adapter])
        _install_materialize_mode(module.lora_B[adapter])
        module._axolotl_deepspeed_native_orig_forward = module.forward
        module._ma_orig_forward = module.forward
        module.forward = types.MethodType(_deepspeed_native_forward, module)
        installed += 1
    return installed


def preserve_deepspeed_native_nvfp4_lora_forwards(model: torch.nn.Module) -> int:
    """Keep PEFT's forward before later kernel patches replace it."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    protected = 0
    for module in getattr(model, "modules", lambda: ())():
        if not isinstance(module, LoraLinear) or hasattr(module, "_ma_orig_forward"):
            continue
        if _native_recipe(module.get_base_layer()) is None:
            continue
        module._ma_orig_forward = module.forward
        protected += 1
    return protected


class DeepSpeedNativeNVFP4MergeAwareCallback(TrainerCallback):
    """Install static native LoRA forwards after Accelerate creates DeepSpeed."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        trainer = self.trainer
        engine = getattr(trainer, "model_wrapped", None)
        model = getattr(engine, "module", None)
        if model is None or not getattr(
            model, "_axolotl_native_nvfp4_deepspeed_merge_aware_requested", False
        ):
            return control
        installed = install_deepspeed_native_nvfp4_merge_aware_lora_linears(model)
        if not installed:
            _unsupported_native_merge_aware(
                model, "no supported static native LoRA projections"
            )
        model._axolotl_native_nvfp4_deepspeed_merge_aware_installed = installed
        return control
