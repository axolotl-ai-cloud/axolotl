"""
Hijack the LlamaAttention forward method to use xformers if available.

Updated for transformers v4.50.0.
"""

from typing import Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

try:
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


def xformers_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    Implements xformers memory-efficient attention for LlamaAttention with support for GQA.

    Args:
        module: The LlamaAttention module
        query: Query states of shape [batch, num_heads, seq_len, head_dim]
        key: Key states of shape [batch, num_kv_heads, seq_len, head_dim]
        value: Value states of shape [batch, num_kv_heads, seq_len, head_dim]
        attention_mask: Attention mask
        scaling: Scaling factor for attention scores
        dropout: Dropout probability

    Returns:
        attn_output: Output of xformers memory-efficient attention
        attn_weights: None
    """
    # First, handle grouped-query attention (GQA)
    # We need to repeat key and value states to match the number of query heads
    num_key_value_groups = getattr(module, "num_key_value_groups", 1)
    key = repeat_kv(key, num_key_value_groups)
    value = repeat_kv(value, num_key_value_groups)

    # xformers expects inputs in shape [batch, seq_len, num_heads, head_dim]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Determine if we need a causal mask
    is_causal = getattr(module, "is_causal", True)

    # Set up the attention bias for xformers
    if is_causal:
        # Use xformers built-in causal mask
        attn_bias = xformers.ops.LowerTriangularMask()
    elif attention_mask is not None:
        # For non-causal attention with a mask, we'd need to convert the mask
        # This is a simplification - you might need to adapt based on your mask format
        attn_bias = attention_mask
    else:
        # No mask needed
        attn_bias = None

    # Apply xformers memory-efficient attention
    attn_output = xformers.ops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout if module.training else 0.0,
        scale=scaling,
    )

    # Reshape back to [batch, seq_len, hidden_size]
    attn_output = attn_output.transpose(1, 2)

    return attn_output, None  # Return None for attn_weights to match interface


def hijack_llama_attention():
    """
    Patch the LlamaAttention forward method to use xformers if available.
    """
    if not XFORMERS_AVAILABLE:
        raise ValueError(
            "xformers not available. Please install it following axolotl's requirements."
        )

    import transformers.models.llama.modeling_llama as llama_modeling

    # Add xformers to the available attention implementations
    llama_modeling.ALL_ATTENTION_FUNCTIONS["xformers"] = xformers_attention_forward

    # Create a wrapper for the original LlamaAttention forward method
    original_forward = llama_modeling.LlamaAttention.forward

    def patched_forward(self, *args, **kwargs):
        # Set the attention implementation to xformers
        # pylint: disable=protected-access
        self.config._attn_implementation = "xformers"
        return original_forward(self, *args, **kwargs)

    # Apply the patch
    llama_modeling.LlamaAttention.forward = patched_forward
