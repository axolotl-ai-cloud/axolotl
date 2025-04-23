"""GLM 4 patch. GLM family inherits from Llama."""

from types import MethodType

import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
)


def patch_glm(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:

    # Set the _PATCH_OPTS in the llama patch file
    import cut_cross_entropy.transformers.llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options  # pylint: disable=protected-access

    from cut_cross_entropy.transformers.llama import cce_forward
    from transformers.models.glm import modeling_glm

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_glm.GlmForCausalLM
        ), f"Expected a GlmForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_glm.GlmForCausalLM.forward = cce_forward
    return None


def patch_glm4(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:

    # Set the _PATCH_OPTS in the llama patch file
    import cut_cross_entropy.transformers.llama as llama_patch

    llama_patch._PATCH_OPTS = patch_options  # pylint: disable=protected-access

    from cut_cross_entropy.transformers.llama import cce_forward
    from transformers.models.glm4 import modeling_glm4

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_glm4.Glm4ForCausalLM
        ), f"Expected a Glm4ForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_glm4.Glm4ForCausalLM.forward = cce_forward
    return None
