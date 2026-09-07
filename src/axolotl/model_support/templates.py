"""Reusable model-family templates for the vanilla loading path."""

from .profile import ModelFamilyTemplate, ModelStrategies


def _causal_lm_auto_model_cls() -> type:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _image_text_to_text_auto_model_cls() -> type:
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


VANILLA_CAUSAL_LM = ModelFamilyTemplate(
    name="vanilla_causal_lm",
    strategies=ModelStrategies(auto_model_cls=_causal_lm_auto_model_cls),
)

IMAGE_TEXT_TO_TEXT = ModelFamilyTemplate(
    name="image_text_to_text",
    is_multimodal=True,
    strategies=ModelStrategies(auto_model_cls=_image_text_to_text_auto_model_cls),
)
