"""Gemma CCE patch"""

# Gemma uses the same forward as Llama

from types import MethodType

import transformers
from cut_cross_entropy.transformers.llama import cce_forward
from cut_cross_entropy.utils import PatchOptions, TransformersModelT

_PATCH_OPTS: PatchOptions | None = None


def patch_gemma(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.gemma import modeling_gemma

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_gemma.GemmaForCausalLM
        ), f"Expected a GemmaForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_gemma.GemmaForCausalLM.forward = cce_forward
    return None
