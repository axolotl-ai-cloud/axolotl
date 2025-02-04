"""
Shared attention helpers
"""

import torch


# Copied from transformers.models.mistral.modeling_mistral (llama.modeling_llama at v4.36)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from:
       (batch, num_key_value_heads, seqlen, head_dim) to
       (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def mask_attention(
    qk_dot: torch.Tensor, attn_mask: torch.Tensor, mask_value: float = -10000
) -> torch.Tensor:
    """
    Apply attention mask (e.g., for padding)
    """
    if len(attn_mask.shape) == 4:  # attn_mask either (b, h, l, d) or (b, l)
        return qk_dot.masked_fill(~attn_mask.bool(), mask_value)
    else:
        return qk_dot.masked_fill(~attn_mask[:, None, None, :].bool(), mask_value)
