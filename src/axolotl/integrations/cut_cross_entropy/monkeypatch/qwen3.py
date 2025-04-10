"""Qwen3 CCE patch. The model inherits Llama's modeling code and uses the same forward method."""

# pylint: disable=duplicate-code

from types import MethodType

import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
)

_PATCH_OPTS: PatchOptions | None = None


def patch_qwen3(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.qwen3 import modeling_qwen3

    from axolotl.integrations.cut_cross_entropy.monkeypatch.llama import cce_forward

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_qwen3.Qwen3ForCausalLM
        ), f"Expected a Qwen3ForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_qwen3.Qwen3ForCausalLM.forward = cce_forward
    return None
