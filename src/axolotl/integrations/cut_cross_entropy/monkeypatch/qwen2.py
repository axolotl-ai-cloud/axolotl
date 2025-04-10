"""Qwen2 CCE patch. The model inherits Llama's modeling code and uses the same forward method."""

# pylint: disable=duplicate-code

from types import MethodType

import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
)

_PATCH_OPTS: PatchOptions | None = None


def patch_qwen2(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.qwen2 import modeling_qwen2

    from axolotl.integrations.cut_cross_entropy.monkeypatch.llama import cce_forward

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_qwen2.Qwen2ForCausalLM
        ), f"Expected a Qwen2ForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_qwen2.Qwen2ForCausalLM.forward = cce_forward
    return None
