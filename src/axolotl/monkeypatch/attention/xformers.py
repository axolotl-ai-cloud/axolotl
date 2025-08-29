"""
xformers attention implementation for packing
"""

from typing import Optional

import torch
import xformers
import xformers.ops.fmha
from transformers.modeling_flash_attention_utils import (
    _upad_input,
)

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids

xformers_attention = xformers.ops.fmha.memory_efficient_attention


def xformers_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    **kwargs,
):
    # Get dimensions
    # query: [batch, heads, seq_len, hidden_dim]
    batch_size = query.size(0)
    query_length = query.shape[2]
    key_length = key.shape[2]

    # Default causal mask
    attn_bias = xformers.ops.LowerTriangularMask()

    # Check if we have sliding window attention
    has_sliding_window = sliding_window is not None and sliding_window < query_length

    # Transpose dimensions for xformers (Q: [b, h, s, d] -> [b, s, h, d])
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Get GQA parameters
    num_attention_heads = module.config.num_attention_heads
    num_key_value_heads = module.config.num_key_value_heads
    head_dim = query.size(-1)
    is_gqa = num_attention_heads != num_key_value_heads
    n_groups = num_attention_heads // num_key_value_heads if is_gqa else 1

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    if position_ids is not None and (
        max_length_q is not None
        or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            cu_seq_lens_q = get_cu_seqlens_from_pos_ids(position_ids)[0]
            cu_seq_lens_q = cu_seq_lens_q.squeeze()
            seq_lengths = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            attn_bias = (
                xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                    q_seqlen=seq_lengths.tolist(),
                )
            )
        else:
            query = query.reshape(-1, query.size(-2), query.size(-1))
            key = key.reshape(-1, key.size(-2), key.size(-1))
            value = value.reshape(-1, value.size(-2), value.size(-1))

        # Handle GQA
        if is_gqa:
            key = key.repeat_interleave(n_groups, dim=2)
            value = value.repeat_interleave(n_groups, dim=2)

    elif attention_mask is not None:
        query, key, value, _, cu_seq_lens, _ = _upad_input(
            query, key, value, attention_mask, query_length
        )
        cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
        seq_lengths = []
        for i in range(len(cu_seq_lens_q) - 1):
            seq_lengths.append(cu_seq_lens_q[i + 1] - cu_seq_lens_q[i])
        attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
            q_seqlen=seq_lengths,
            kv_seqlen=seq_lengths,
        )

        # Handle GQA
        if is_gqa:
            key = key.repeat_interleave(n_groups, dim=2)
            value = value.repeat_interleave(n_groups, dim=2)
    else:
        # Handle Group Query Attention (GQA) using view/expand approach from reference
        key = key.view(batch_size, key_length, num_key_value_heads, 1, head_dim)
        value = value.view(batch_size, key_length, num_key_value_heads, 1, head_dim)
        key = key.expand(
            batch_size, key_length, num_key_value_heads, n_groups, head_dim
        )
        value = value.expand(
            batch_size, key_length, num_key_value_heads, n_groups, head_dim
        )

        if module.training:
            key = key.reshape(batch_size, key_length, num_attention_heads, head_dim)
            value = value.reshape(batch_size, key_length, num_attention_heads, head_dim)

            if has_sliding_window:
                query = query.view(
                    1, batch_size * query_length, num_attention_heads, head_dim
                )
                key = key.view(
                    1, batch_size * key_length, num_attention_heads, head_dim
                )
                value = value.view(
                    1, batch_size * key_length, num_attention_heads, head_dim
                )
        else:
            query = query.view(
                batch_size, query_length, num_key_value_heads, n_groups, head_dim
            )

            # If we need a sliding window attention
            if has_sliding_window:
                query = query.view(
                    1,
                    batch_size * query_length,
                    num_key_value_heads,
                    n_groups,
                    head_dim,
                )
                key = key.view(
                    1, batch_size * key_length, num_key_value_heads, n_groups, head_dim
                )
                value = value.view(
                    1, batch_size * key_length, num_key_value_heads, n_groups, head_dim
                )

    # Run the xformers attention
    attn_output = xformers_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
    )

    attn_output = attn_output.view(
        batch_size, -1, attn_output.size(-2), attn_output.size(-1)
    )
    return attn_output, None
