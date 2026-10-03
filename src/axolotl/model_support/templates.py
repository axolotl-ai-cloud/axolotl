"""Reusable model-family templates for the vanilla loading path."""

from .base import Unsupported
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

DIFFUSION_LM = ModelFamilyTemplate(
    name="diffusion_lm",
    capabilities={
        "cut_cross_entropy": Unsupported(
            "Native diffusion computes its objective from aligned canvas logits."
        ),
        "liger": Unsupported("Native diffusion has no Liger kernel validation."),
        "context_parallel": Unsupported(
            "Native diffusion document and canvas masks have no context-parallel validation."
        ),
        "expert_kernels": Unsupported(
            "Native diffusion has no expert-kernel training validation."
        ),
    },
)
