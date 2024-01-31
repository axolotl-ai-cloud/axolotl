"""
Patched LlamaAttention to use torch.nn.functional.scaled_dot_product_attention
"""

from typing import Optional

import torch
import transformers.models.llama.modeling_llama
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.monkeypatch.utils import mask_2d_to_4d


def patched_prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *args,
):
    dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float32
    return _prepare_4d_causal_attention_mask(
        mask_2d_to_4d(attention_mask, dtype=dtype),
        *args,
    )


def patched_prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    *args,
):
    dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float32
    return _prepare_4d_causal_attention_mask_for_sdpa(
        mask_2d_to_4d(attention_mask, dtype=dtype),
        *args,
    )


def hijack_llama_prepare_4d_mask():
    transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask_for_sdpa
    )
    transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask
    )
    transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask
    )
