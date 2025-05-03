"""
xformers attention implementation for packing
"""

from typing import Optional

import torch
import xformers
import xformers.ops.fmha
from flash_attn.bert_padding import pad_input
from transformers.modeling_flash_attention_utils import (
    _upad_input,
    prepare_fa2_from_position_ids,
)

xformers_attention = xformers.ops.fmha.memory_efficient_attention


def xformers_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    dropout: float = 0.0,  # pylint: disable=unused-argument
    scaling: Optional[float] = None,  # pylint: disable=unused-argument
    sliding_window: Optional[int] = None,  # pylint: disable=unused-argument
    softcap: Optional[float] = None,  # pylint: disable=unused-argument
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,  # pylint: disable=unused-argument
    **kwargs,  # pylint: disable=unused-argument
):

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_length = query.shape[2]

    attn_bias = xformers.ops.LowerTriangularMask()

    if attention_mask is not None:
        batch_size = query.shape[0]
        query, key, value, indices_q, cu_seq_lens, _ = _upad_input(
            query, key, value, attention_mask, query_length
        )
        cu_seqlens_q, cu_seq_lens_k = cu_seq_lens
        seq_lengths = []
        for i in range(len(cu_seq_lens_q) - 1):
            seq_lengths.append(cu_seqlens_q[i + 1] - cu_seq_lens_q[i])
        attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
            q_seqlen=seq_lengths,
            kv_seqlen=seq_lengths,
        )

        attn_output_unpad = xformers_attention(
            query,
            key,
            value,
            attn_bias=attn_bias,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif position_ids is not None and (
        max_length_q is not None
        or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        batch_size = query.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            _, _, _, indices_q, cu_seq_lens, _ = prepare_fa2_from_position_ids(
                query, key, value, position_ids
            )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            seq_lengths = []
            for i in range(len(cu_seq_lens_q) - 1):
                seq_lengths.append(
                    cu_seq_lens_q[i + 1].item() - cu_seq_lens_q[i].item()
                )
            attn_bias = (
                xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                    q_seqlen=seq_lengths,
                )
            )

        else:
            query = query.reshape(-1, query.size(-2), query.size(-1))
            key = key.reshape(-1, key.size(-2), key.size(-1))
            value = value.reshape(-1, value.size(-2), value.size(-1))

        if module.config.num_attention_heads != module.config.num_key_value_heads:
            key = key.repeat_interleave(
                module.config.num_attention_heads // module.config.num_key_value_heads,
                dim=2,
            )
            value = value.repeat_interleave(
                module.config.num_attention_heads // module.config.num_key_value_heads,
                dim=2,
            )

            attn_output = xformers_attention(
                query,
                key,
                value,
                attn_bias=attn_bias,
            )

        else:
            attn_output = xformers_attention(
                query,
                key,
                value,
                attn_bias=attn_bias,
            )

    else:
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
