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
    batch_size = query.size(0)

    attn_bias = xformers.ops.LowerTriangularMask()

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

        # pylint: disable=duplicate-code
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
    elif attention_mask is not None:
        query, key, value, indices_q, cu_seq_lens, _ = _upad_input(
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

        attn_output_unpad = xformers_attention(
            query,
            key,
            value,
            attn_bias=attn_bias,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        # pylint: disable=duplicate-code
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

    attn_output = attn_output.view(
        batch_size, -1, attn_output.size(-2), attn_output.size(-1)
    )
    return attn_output, None
