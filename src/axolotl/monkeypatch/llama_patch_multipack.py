"""
Patched LlamaAttention to use torch.nn.functional.scaled_dot_product_attention
"""

from axolotl.monkeypatch.utils import (
    patched_prepare_4d_causal_attention_mask,
    patched_prepare_4d_causal_attention_mask_for_sdpa,
)


def hijack_llama_prepare_4d_mask():
    from transformers import modeling_attn_mask_utils
    from transformers.models.llama import modeling_llama

    modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask_for_sdpa
    )
    modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask_for_sdpa
    )
    modeling_llama._prepare_4d_causal_attention_mask = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask
    )
    modeling_attn_mask_utils._prepare_4d_causal_attention_mask = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask
    )
