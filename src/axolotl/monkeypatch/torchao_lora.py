"""Training eligibility for frozen native NVFP4 weights with PEFT adapters."""

from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer


class _NVFP4LoRAQuantizer(TorchAoHfQuantizer):
    @property
    def is_trainable(self):
        return True


def enable_native_nvfp4_lora_training(model):
    """Relax only this model's quantizer after checking the supported base layout."""
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
    if any(p.act_quant_kwargs is not None for p in weights) and not getattr(
        model, "_axolotl_native_nvfp4_zero3_dynamic_allowed", False
    ):
        return False
    quantizer.__class__ = _NVFP4LoRAQuantizer
    return True
