"""Training eligibility for frozen native NVFP4 weights with PEFT adapters."""

from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer


class _NVFP4LoRAQuantizer(TorchAoHfQuantizer):
    @property
    def is_trainable(self):
        return True


def _has_dynamic_native_nvfp4_input_gradient_path(model) -> bool:
    if getattr(model, "_axolotl_native_nvfp4_zero3_dynamic_allowed", False):
        return True
    if getattr(model, "_axolotl_native_nvfp4_dynamic_input_gradients", False):
        return True
    if getattr(
        model, "_axolotl_native_nvfp4_dynamic_input_gradients_requested", None
    ) in ("FSDP", "DeepSpeed"):
        from axolotl.monkeypatch.torchao_nvfp4_dynamic_ste import (
            native_nvfp4_dynamic_input_ste_preflight,
        )

        return native_nvfp4_dynamic_input_ste_preflight(model)
    return False


def enable_native_nvfp4_lora_training(model):
    """Relax only native NVFP4 bases with a supported LoRA training path."""
    quantizer = getattr(model, "hf_quantizer", None)
    if not isinstance(quantizer, TorchAoHfQuantizer) or not getattr(
        model, "peft_config", None
    ):
        return False
    if type(quantizer.quantization_config.quant_type).__name__ not in {
        "NVFP4WeightOnlyConfig",
        "NVFP4DynamicActivationNVFP4WeightConfig",
    }:
        return False
    weights = [p for p in model.parameters() if type(p).__name__ == "NVFP4Tensor"]
    if not weights or any(p.requires_grad for p in weights):
        return False
    if any(p.act_quant_kwargs is not None for p in weights) and not (
        _has_dynamic_native_nvfp4_input_gradient_path(model)
    ):
        return False
    quantizer.__class__ = _NVFP4LoRAQuantizer
    return True
