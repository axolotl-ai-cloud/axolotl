"""
Flash Attention monkey patch for Falcon

copied from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/falcon_flash_attn_monkey_patch.py
"""

from typing import Optional, Tuple

import torch
import transformers
from flash_attn import flash_attn_func


def forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,  # pylint: disable=unused-argument
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    use_cache: bool = False,
    output_attentions: bool = False,  # pylint: disable=unused-argument
):
    fused_qkv = self.query_key_value(
        hidden_states
    )  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = (
        self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    )
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (
        query_layer,
        key_layer,
        value_layer,
    ) = self._split_heads(  # pylint: disable=protected-access
        fused_qkv
    )

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(
        batch_size * self.num_heads, query_length, self.head_dim
    )
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv_heads,
        query_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(
        batch_size * num_kv_heads, query_length, self.head_dim
    )

    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
    query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, kv_length, head_dim]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    # unused
    # _, kv_length, _ = key_layer.shape
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None
    # unused
    # attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float("-1e9")).to(query_layer.dtype)
    query_layer_ = (
        query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        .transpose(1, 2)
        .to(torch.bfloat16)
    )
    key_layer_ = (
        key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        .transpose(1, 2)
        .to(torch.bfloat16)
    )
    value_layer_ = (
        value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        .transpose(1, 2)
        .to(torch.bfloat16)
    )

    if alibi is not None:
        raise ValueError("`alibi` is not supported when `use_flash_attn` is True")

    # below output will have shape (batch_size, seqlen, nheads, headdim)
    attn_output = flash_attn_func(query_layer_, key_layer_, value_layer_, causal=True)
    attn_output = attn_output.reshape(
        batch_size, query_length, self.num_heads * self.head_dim
    )
    output_tensor = self.dense(attn_output)
    return output_tensor, present


def replace_falcon_attn_with_flash_attn():
    transformers.models.falcon.modeling_falcon.FalconAttention.forward = forward
