"""Inkling model support (hybrid sliding/full attention with short convolutions)."""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT, VANILLA_CAUSAL_LM


def _before_model_build(context: ModelHookContext) -> None:
    if context.cfg.sample_packing or context.cfg.batch_flattening:
        from .modeling import patch_inkling_modeling_packing

        patch_inkling_modeling_packing()


_HOOKS = ModelHooks(
    by_phase={ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,)}
)


@register_model_support
class InklingSupport(ModelSupport):
    """Descriptor for multimodal Inkling (`InklingForConditionalGeneration`)."""

    model_types = ("inkling_mm_model",)
    profile = ModelProfile(family=IMAGE_TEXT_TO_TEXT, hooks=_HOOKS)


@register_model_support
class InklingTextSupport(ModelSupport):
    """Descriptor for text-only Inkling (`InklingForCausalLM`)."""

    model_types = ("inkling_text",)
    profile = ModelProfile(family=VANILLA_CAUSAL_LM, hooks=_HOOKS)
