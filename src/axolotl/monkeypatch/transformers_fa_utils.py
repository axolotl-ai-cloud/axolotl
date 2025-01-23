"""
see https://github.com/huggingface/transformers/pull/35834
"""

import logging
from functools import partial
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def fixed_fa_peft_integration_check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
    preferred_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!

    Args:
        query (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        target_dtype (`torch.dtype`, *optional*):
            The dtype to convert the attention tensors to. Conversion can be ignored by
            not providing the target dtype.
        preferred_dtype (`torch.dtype`, *optional*):
            The preferred dtype to convert the attention tensors to regardless of the
            target dtype.
    """
    if target_dtype is None and preferred_dtype is None:
        return query, key, value

    if preferred_dtype and target_dtype != preferred_dtype:
        target_dtype = preferred_dtype

    # check if any of query, key, or value are in float32. If so, cast them back to target dtype.
    if any(module.dtype == torch.float32 for module in [query, key, value]):
        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    return query, key, value


def patch_fa_peft_integration():
    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils.fa_peft_integration_check = partial(
        fixed_fa_peft_integration_check, preferred_dtype=None
    )
