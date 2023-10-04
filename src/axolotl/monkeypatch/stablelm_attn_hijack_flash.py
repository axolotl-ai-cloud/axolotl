# coding=utf-8
# Copyright 2023 Stability AI, EleutherAI, and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is based off the following work:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
""" PyTorch StableLM Epoch model. """
import importlib
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from accelerate import init_empty_weights
from flash_attn.flash_attn_interface import flash_attn_func
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


def replace_stablelm_attn_with_flash_attn(model_name="stabilityai/stablelm-3b-4e1t"):
    # this is a wonky hack to get the remotely loaded module
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # we need to load the model here in order for modeling_btlm to be available
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    module_name = model_config.__class__.__module__.replace(
        ".configuration_stablelm_epoch", ".modeling_stablelm_epoch"
    )
    modeling_stablelm = importlib.import_module(module_name)
    modeling_stablelm.Attention.forward = (  # pylint: disable=protected-access
        flashattn_attn
    )


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def flashattn_attn(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

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

    query_rot = query_states[..., : self.rotary_ndims]
    query_pass = query_states[..., self.rotary_ndims :]
    key_rot = key_states[..., : self.rotary_ndims]
    key_pass = key_states[..., self.rotary_ndims :]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_rot, key_rot, cos, sin, position_ids
    )

    # [batch_size, num_heads, seq_len, head_dim]
    query_states = torch.cat((query_states, query_pass), dim=-1)
    key_states = torch.cat((key_states, key_pass), dim=-1)

    if past_key_value is not None:
        # Reuse k, v, self_attention
        key_states = torch.cat((past_key_value[0], key_states), dim=2)
        value_states = torch.cat((past_key_value[1], value_states), dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Prepare q and kv for flash attention
    # q = query_states
    # kv = torch.stack([key_states, value_states], dim=1)

    softmax_scale = 1.0 / math.sqrt(self.head_dim)

    # Call flash attention function
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout_p=0.0,  # Assuming you have this attribute
        softmax_scale=softmax_scale,  # Set this if you have specific scaling in mind
        causal=True,  # Assuming you have this attribute
        return_attn_probs=False,  # Set this based on your needs
    )
    #
    # attn_output, _, _ = flash_attn_varlen_kvpacked_func(
    #     q=q,
    #     kv=kv,
    #     cu_seqlens_q=cu_seqlens_q,
    #     cu_seqlens_k=cu_seqlens_k,
    #     max_seqlen_q=max_seqlen_q,
    #     max_seqlen_k=max_seqlen_k,
    #     dropout_p=self.config.attention_dropout,
    #     softmax_scale=1.0 / math.sqrt(self.head_dim),
    #     causal=self.config.is_decoder,
    # )
    #

    # Merge heads
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # Final linear projection
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
