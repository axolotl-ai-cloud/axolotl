"""
Directly copied the code from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py and made some adjustments
"""

import logging
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    import xformers.ops
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")


def hijack_llama_attention():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # pylint: disable=duplicate-code
    bsz, q_len, _ = hidden_states.size()

    if not hasattr(self, "pretraining_tp"):
        self.pretraining_tp = 1

    if self.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    #
    # xformers-attn start
    #

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
    # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
    if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
        # input and output should be of form (bsz, q_len, num_heads, head_dim)
        attn_output = xformers.ops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=None
        )
    else:
        # input and output should be of form (bsz, q_len, num_heads, head_dim)
        attn_output = xformers.ops.memory_efficient_attention(
            query_states,
            key_states,
            value_states,
            # attn_bias=attention_mask,
            attn_bias=xformers.ops.LowerTriangularMask(),
        )

    if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    #
    # xformers-attn end
    #

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.pretraining_tp, dim=1
        )
        attn_output = sum(
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.pretraining_tp)
        )
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
