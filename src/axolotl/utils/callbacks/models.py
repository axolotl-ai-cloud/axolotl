"""Helper functions for model classes"""

from typing import Tuple

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def get_causal_lm_model_cls_prefix(model_type: str) -> Tuple[str, str]:
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        causal_lm_cls = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]
        causal_lm_cls_prefix = causal_lm_cls
        for suffix in [
            "ForCausalLM",
            "ForConditionalGeneration",
            "LMHeadModel",
            "GenerationDecoder",
        ]:
            causal_lm_cls_prefix = causal_lm_cls_prefix.replace(suffix, "")
        return causal_lm_cls_prefix, causal_lm_cls
    causal_lm_cls_prefix = "".join(
        [part.capitalize() for part in model_type.split("_")]
    )
    return causal_lm_cls_prefix, f"{causal_lm_cls_prefix}ForCausalLM"
