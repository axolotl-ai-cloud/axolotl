"""Reusable model-family templates for the vanilla loading path."""

from typing import Any

from .base import Unsupported
from .diffusion import is_native_diffusion
from .profile import ModelFamilyTemplate, ModelStrategies


def _causal_lm_auto_model_cls() -> type:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _image_text_to_text_auto_model_cls() -> type:
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _diffusion_lm_trainer_cls(cfg: Any) -> type | None:
    if not is_native_diffusion(cfg):
        return None
    from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer

    return AxolotlDiffusionTrainer


def _diffusion_lm_collator_factory(cfg: Any, tokenizer: Any, is_eval: bool) -> Any:
    if not is_native_diffusion(cfg):
        return None
    from axolotl.core.trainers.diffusion_lm.collator import build_diffusion_collator

    return build_diffusion_collator(cfg, tokenizer, is_eval=is_eval)


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
    strategies=ModelStrategies(
        trainer_cls=_diffusion_lm_trainer_cls,
        collator_factory=_diffusion_lm_collator_factory,
    ),
)
