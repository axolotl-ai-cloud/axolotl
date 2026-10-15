"""Limit PEFT's Mamba exception to models with adapter-aware projections."""

from functools import wraps


def enable_lora_projections():
    from peft.tuners import tuners_utils

    original = tuners_utils._check_lora_target_modules_mamba
    if getattr(original, "_axolotl_fla", False):
        return

    @wraps(original)
    def check(peft_config, model, target_name):
        layers = getattr(getattr(model, "backbone", None), "layers", ())
        if (
            layers
            and all(hasattr(block.mixer, "fla_mixer") for block in layers)
            and peft_config.peft_type == "LORA"
        ):
            if target_name in ("in_proj", "out_proj"):
                return
            raise ValueError("FLA Mamba LoRA targets must be in_proj or out_proj")
        return original(peft_config, model, target_name)

    check._axolotl_fla = True
    tuners_utils._check_lora_target_modules_mamba = check
