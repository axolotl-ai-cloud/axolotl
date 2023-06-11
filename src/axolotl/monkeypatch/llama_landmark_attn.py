# pylint: skip-file
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""
PyTorch LLaMA model.
Taken from https://github.com/epfml/landmark-attention/blob/main/llama/llama_mem.py and modified.
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    _expand_mask,
    _make_causal_mask,
    rotate_half,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

MEM_TOKEN = "<landmark>"  # nosec


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    if q is None:
        q_embed = None
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LandmarkGroupedSoftmaxFunction(torch.autograd.Function):
    """
    Landmark grouped softmax function.
    """

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, dim, mem_cnt, resp_mem_idx):
        new_shape = list(x.shape)
        new_shape[dim] = mem_cnt  # max_mem_cnt.item()
        max_by_group = x.new_zeros((*new_shape,))
        max_by_group.scatter_reduce_(
            src=x, index=resp_mem_idx, dim=dim, reduce="amax", include_self=False
        )

        maxes = torch.gather(max_by_group, dim, resp_mem_idx)
        # x_exp = torch.exp(x - torch.where(torch.isinf(maxes), 0, maxes))
        x_exp = torch.exp((x - maxes).to(torch.float32))

        cumsum_by_group = torch.zeros_like(max_by_group, dtype=x_exp.dtype)

        cumsum_by_group.scatter_add_(
            dim,
            resp_mem_idx,
            x_exp,
        )
        denom = torch.gather(cumsum_by_group, dim, resp_mem_idx)

        # probs = torch.where(denom < 0.5, 0, x_exp / denom)
        probs = x_exp / denom

        ctx.mem_cnt = mem_cnt
        ctx.dim = dim
        ctx.save_for_backward(resp_mem_idx, probs)

        return probs

    @staticmethod
    def backward(ctx, grad_probs):
        mem_cnt = ctx.mem_cnt
        dim = ctx.dim
        resp_mem_idx, probs = ctx.saved_tensors
        grad_x = grad_dim = grad_mem_cnt = grad_resp_mem_idx = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[4]:
            grad_pair = grad_probs * probs

            new_shape = list(probs.shape)
            new_shape[dim] = mem_cnt  # max_mem_cnt.item()
            cumsum_by_group = grad_pair.new_zeros((*new_shape,))
            cumsum_by_group.scatter_add_(dim, resp_mem_idx, grad_pair)

        if ctx.needs_input_grad[0]:
            grad_sum = torch.gather(cumsum_by_group, dim, resp_mem_idx)
            grad_x = grad_pair - probs * grad_sum
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]
        assert not ctx.needs_input_grad[3]

        return grad_x, grad_dim, grad_mem_cnt, grad_resp_mem_idx


def landmark_grouped_softmax(x, dim, is_mem, last_section_mask):
    last_and_rest_mask = last_section_mask  # | mask

    full_access_mask = is_mem | last_and_rest_mask

    max_mem_cnt = 16
    mem_group_idx = torch.cumsum(is_mem, dim=dim)
    mem_bucket_id = max_mem_cnt - 1
    resp_mem_idx = torch.where(
        last_and_rest_mask,
        max_mem_cnt - 1,
        torch.where(is_mem, mem_bucket_id, mem_group_idx),
    )
    probs = LandmarkGroupedSoftmaxFunction.apply(x, dim, max_mem_cnt, resp_mem_idx)

    new_shape = list(x.shape)
    new_shape[dim] = max_mem_cnt
    group_prob = probs.new_zeros((*new_shape,))
    group_prob.scatter_(
        dim, torch.where(is_mem, mem_group_idx - 1, max_mem_cnt - 1), probs
    )
    probs = probs.mul(
        torch.where(
            full_access_mask,
            last_section_mask,
            torch.gather(group_prob, dim, resp_mem_idx),
        )
    )

    return probs


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings
        )

        self.mem_freq = None
        self.top_k = None
        self.max_cache_size = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def set_mem_cache_args(self, mem_freq, top_k, max_cache_size):
        self.mem_freq = mem_freq
        self.top_k = top_k
        self.max_cache_size = max_cache_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        is_mem: Optional[torch.Tensor] = None,
        last_section_mask: Optional[torch.Tensor] = None,
        offload_cache_to_cpu: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            if len(past_key_value) > 2:
                kv_seq_len += past_key_value[3].shape[2] * past_key_value[3].shape[3]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        key_states_before_pos = key_states
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]

        attn_prefix = None
        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.mem_freq is None:
                cache_len = past_key_value[0].shape[2]
                if self.max_cache_size is not None:
                    cache_len = min(cache_len, self.max_cache_size)
                if is_mem is not None:
                    is_mem = torch.cat(
                        (is_mem.new_zeros((1, 1, q_len, cache_len)), is_mem), dim=-1
                    )
                    last_section_mask = torch.cat(
                        (
                            last_section_mask.new_ones((1, 1, q_len, cache_len)),
                            last_section_mask,
                        ),
                        dim=-1,
                    )

                past_key_states = torch.cat([past_key_value[0], key_states], dim=2)
                past_value_states = torch.cat([past_key_value[1], value_states], dim=2)
                key_states = past_key_states[:, :, -(q_len + cache_len) :]
                value_states = past_value_states[:, :, -(q_len + cache_len) :]
                expected_att_size = (bsz, self.num_heads, q_len, cache_len + q_len)
            else:
                orig_value_states = value_states

                incomplete_len = past_key_value[0].shape[2] % (self.mem_freq + 1)
                full_len = past_key_value[0].shape[2] - incomplete_len
                past_key_mem, past_key_incomplete = torch.split(
                    past_key_value[0], (full_len, incomplete_len), dim=2
                )
                past_value_mem, past_value_incomplete = torch.split(
                    past_key_value[1], (full_len, incomplete_len), dim=2
                )

                if offload_cache_to_cpu:
                    past_key_value = (
                        past_key_incomplete,
                        past_value_incomplete,
                        *past_key_value[2:],
                    )

                if incomplete_len > 0:
                    assert q_len + incomplete_len <= (self.mem_freq + 1)
                is_mem = torch.cat(
                    (is_mem.new_zeros((1, 1, q_len, incomplete_len)), is_mem), dim=-1
                )
                last_section_mask = torch.cat(
                    (
                        last_section_mask.new_ones((1, 1, q_len, incomplete_len)),
                        last_section_mask,
                    ),
                    dim=-1,
                )

                if len(past_key_value) > 2:
                    full_len += past_key_value[3].shape[2] * past_key_value[3].shape[3]
                past_key_incomplete_pos = torch.arange(
                    full_len,
                    full_len + incomplete_len,
                    dtype=torch.long,
                    device=position_ids.device,
                ).unsqueeze(0)
                _, past_key_incomplete = apply_rotary_pos_emb(
                    None, past_key_incomplete, cos, sin, past_key_incomplete_pos
                )
                key_states = torch.cat((past_key_incomplete, key_states), dim=2)
                value_states = torch.cat((past_value_incomplete, value_states), dim=2)

                past_key_mem = past_key_mem.view(
                    bsz, self.num_heads, -1, self.mem_freq + 1, self.head_dim
                )
                past_value_mem = past_value_mem.view(
                    bsz, self.num_heads, -1, self.mem_freq + 1, self.head_dim
                )

                if len(past_key_value) > 2:
                    mem_key_nopos = torch.cat(
                        (
                            past_key_value[2],
                            past_key_mem.select(dim=3, index=self.mem_freq),
                        ),
                        dim=2,
                    )
                    past_key_mem_offload = past_key_value[3]
                    past_key_mem = torch.cat(
                        (
                            past_key_mem_offload,
                            past_key_mem.to(past_key_mem_offload.device),
                        ),
                        dim=2,
                    )
                    past_value_mem = torch.cat(
                        (
                            past_key_value[4],
                            past_value_mem.to(past_key_mem_offload.device),
                        ),
                        dim=2,
                    )
                else:
                    mem_key_nopos = past_key_mem.select(dim=3, index=self.mem_freq)

                num_mems = past_key_mem.shape[2]
                top_k = min(self.top_k, num_mems)
                prefix_len = full_len - (top_k + 1) * (self.mem_freq + 1)
                mem_indices = torch.cat(
                    (
                        position_ids.new_zeros((max(0, num_mems - top_k),)),
                        torch.arange(
                            1,
                            top_k + 1,
                            device=query_states.device,
                            dtype=position_ids.dtype,
                        ),
                    ),
                    dim=0,
                )
                mem_pos = (mem_indices * (self.mem_freq + 1) + self.mem_freq).unsqueeze(
                    0
                ).expand(bsz, -1) + prefix_len
                _, mem_key = apply_rotary_pos_emb(
                    None, mem_key_nopos, cos, sin, mem_pos
                )
                mem_attn_weights = torch.matmul(
                    query_states, mem_key.transpose(2, 3)
                ) / math.sqrt(self.head_dim)

                if offload_cache_to_cpu:
                    aggregate = "max_over_tokens"
                else:
                    aggregate = None
                if aggregate == "max_over_tokens":
                    token_retrievers = 1
                    head_retrievers = self.num_heads
                    mem_attn_weights = torch.nn.functional.softmax(
                        mem_attn_weights, dim=-1
                    )
                    mem_attn_weights = mem_attn_weights.amax(dim=2, keepdim=True)
                elif aggregate is None:
                    token_retrievers = q_len
                    head_retrievers = self.num_heads
                else:
                    raise NotImplementedError()

                mem_selected_idx = (
                    mem_attn_weights.topk(dim=-1, k=top_k)[1]
                    .sort(dim=-1)[0]
                    .view(bsz, head_retrievers, token_retrievers, top_k)
                )

                selected_indices = torch.arange(
                    0,
                    top_k * (self.mem_freq + 1),
                    device=query_states.device,
                    dtype=position_ids.dtype,
                )
                selected_indices = torch.where(
                    mem_selected_idx >= num_mems - top_k, self.mem_freq + 1, 0
                ).unsqueeze(-1) + selected_indices.view(
                    1, 1, 1, top_k, self.mem_freq + 1
                )
                selected_indices = (
                    selected_indices.view(
                        bsz, head_retrievers, token_retrievers, -1
                    ).expand(bsz, self.num_heads, q_len, -1)
                    + prefix_len
                )

                mem_selected_idx = mem_selected_idx.to(past_key_mem.device)

                mem_selected_idx = mem_selected_idx.view(
                    bsz, self.num_heads, token_retrievers, top_k, 1, 1
                ).expand(
                    bsz,
                    self.num_heads,
                    token_retrievers,
                    top_k,
                    self.mem_freq + 1,
                    self.head_dim,
                )
                selected_keys = past_key_mem.unsqueeze(2).expand(
                    bsz,
                    self.num_heads,
                    token_retrievers,
                    -1,
                    self.mem_freq + 1,
                    self.head_dim,
                )
                selected_keys = selected_keys.take_along_dim(
                    mem_selected_idx, dim=3
                ).to(query_states.device)
                selected_values = (
                    past_value_mem.unsqueeze(2)
                    .expand(
                        bsz,
                        self.num_heads,
                        token_retrievers,
                        -1,
                        self.mem_freq + 1,
                        self.head_dim,
                    )
                    .take_along_dim(mem_selected_idx, dim=3)
                    .to(query_states.device)
                )

                selected_keys = selected_keys.view(
                    bsz, self.num_heads, token_retrievers, -1, self.head_dim
                ).expand(bsz, self.num_heads, q_len, -1, self.head_dim)
                selected_keys = apply_rotary_pos_emb(
                    None, selected_keys.unsqueeze(1), cos, sin, selected_indices
                )[1].squeeze(1)
                selected_values = selected_values.view(
                    bsz, self.num_heads, token_retrievers, -1, self.head_dim
                ).expand(bsz, self.num_heads, q_len, -1, self.head_dim)
                attn_prefix = torch.matmul(
                    query_states.unsqueeze(3), selected_keys.transpose(3, 4)
                ).squeeze(3) / math.sqrt(self.head_dim)
                is_mem_prefix = (
                    torch.cat(
                        (is_mem.new_zeros((self.mem_freq,)), is_mem.new_ones((1,)))
                    )
                    .unsqueeze(0)
                    .repeat((top_k, 1))
                )
                is_mem_prefix = is_mem_prefix.view(1, 1, 1, -1).expand(1, 1, q_len, -1)
                is_mem = torch.cat((is_mem_prefix, is_mem), dim=-1)
                last_section_mask = torch.cat(
                    (
                        last_section_mask.new_zeros(
                            (1, 1, q_len, top_k * (self.mem_freq + 1))
                        ),
                        last_section_mask,
                    ),
                    dim=-1,
                )
                expected_att_size = (bsz, self.num_heads, q_len, q_len + incomplete_len)

                past_key_states = torch.cat(
                    [past_key_value[0], key_states_before_pos], dim=2
                )
                past_value_states = torch.cat(
                    [past_key_value[1], orig_value_states], dim=2
                )

                if offload_cache_to_cpu:
                    past_key_value = (
                        (
                            past_key_states,
                            past_value_states,
                            mem_key_nopos,
                            past_key_mem.to("cpu"),
                            past_value_mem.to("cpu"),
                            *past_key_value[5:],
                        )
                        if use_cache
                        else None
                    )
                else:
                    past_key_value = (
                        (past_key_states, past_value_states) if use_cache else None
                    )

        else:
            if self.mem_freq is None:
                past_key_states = key_states
            else:
                past_key_states = key_states_before_pos
            past_value_states = value_states
            expected_att_size = (bsz, self.num_heads, q_len, kv_seq_len)
            past_key_value = (past_key_states, past_value_states) if use_cache else None

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attn_weights.size() != expected_att_size:
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask[..., -attn_weights.shape[-1] :]
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
        if attn_prefix is not None:
            attn_weights = torch.cat((attn_prefix, attn_weights), dim=-1)
        # upcast attention to fp32
        if is_mem is None:
            raise ValueError("Don't use this without landmarks")

        attn_weights = landmark_grouped_softmax(
            attn_weights,
            dim=-1,
            is_mem=is_mem.expand(-1, self.num_heads, -1, -1),
            last_section_mask=last_section_mask,
        ).to(query_states.dtype)

        if attn_prefix is not None:
            attn_prefix, attn_weights = torch.split(
                attn_weights,
                (attn_prefix.shape[-1], attn_weights.shape[-1] - attn_prefix.shape[-1]),
                dim=-1,
            )
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_prefix is not None:
            attn_output += torch.matmul(
                attn_prefix.unsqueeze(3), selected_values
            ).squeeze(3)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    """
    Llama Decoder layer
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def set_mem_cache_args(self, mem_freq, top_k, max_cache_size):
        self.self_attn.set_mem_cache_args(mem_freq, top_k, max_cache_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        is_mem: Optional[torch.Tensor] = None,
        last_section_mask: Optional[torch.Tensor] = None,
        offload_cache_to_cpu: bool = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            is_mem=is_mem,
            last_section_mask=last_section_mask,
            offload_cache_to_cpu=offload_cache_to_cpu,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mem_id = None

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_mem_id(self, mem_id):
        self.mem_id = mem_id

    def set_mem_cache_args(self, mem_freq, top_k, max_cache_size):
        for layer in self.layers:
            layer.set_mem_cache_args(mem_freq, top_k, max_cache_size)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        offload_cache_to_cpu: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        is_mem = None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            if self.mem_id is not None:
                with torch.no_grad():
                    is_mem = input_ids == self.mem_id
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if self.mem_id is not None:
                raise NotImplementedError
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            if is_mem is not None:
                pass
                # raise NotImplementedError
            past_key_values_length = past_key_values[0][0].shape[2]
            if len(past_key_values[0]) > 2:
                past_key_values_length += (
                    past_key_values[0][3].shape[2] * past_key_values[0][3].shape[3]
                )
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        last_section_mask = None
        if is_mem is not None:
            is_mem = is_mem.unsqueeze(1).unsqueeze(2)
            current_len = input_ids.shape[1]
            mem_ids = torch.where(
                attention_mask[..., -current_len:] < -1,
                0,
                torch.cumsum(is_mem, -1) - is_mem.int(),
            )
            last_section_mask = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
            attention_mask[..., -current_len:].masked_fill_(
                last_section_mask & is_mem,
                torch.tensor(
                    torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device
                ),
            )
            last_section_mask.logical_and_(attention_mask[..., -current_len:] > -1)
            is_mem = is_mem.logical_and(attention_mask[..., -current_len:] > -1)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    None,
                    is_mem,
                    last_section_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    is_mem=is_mem,
                    last_section_mask=last_section_mask,
                    offload_cache_to_cpu=offload_cache_to_cpu,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    """
    Llama model with a causal language modeling head.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mem_id = None
        self.mem_freq = None
        self.top_k = None
        self.max_seq_len = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        offload_cache_to_cpu: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        window_len = self.max_seq_len or input_ids.shape[1]
        last_logits = None
        for _, idx in enumerate(range(0, input_ids.shape[1], window_len)):
            if idx >= 1:
                if output_attentions or output_hidden_states:
                    raise NotImplementedError
                if not use_cache:
                    raise NotImplementedError
            outputs = self.model(
                input_ids=input_ids[:, idx : idx + window_len],
                attention_mask=attention_mask[
                    :, : idx + window_len + attention_mask.shape[1] - input_ids.shape[1]
                ]
                if attention_mask is not None
                else None,
                position_ids=position_ids[:, idx : idx + window_len]
                if position_ids is not None
                else None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, idx : idx + window_len]
                if inputs_embeds is not None
                else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                offload_cache_to_cpu=offload_cache_to_cpu,
            )
            past_key_values = outputs.past_key_values
            if last_logits is not None:
                last_logits = torch.cat((last_logits, outputs[0]), dim=-2)
            last_logits = outputs[0]

        hidden_states = last_logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_mem_id(self, mem_id):
        self.mem_id = mem_id
        self.model.set_mem_id(mem_id)

    def set_mem_cache_args(self, max_seq_len, mem_freq, top_k, max_cache_size):
        self.mem_freq = mem_freq
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        if self.max_seq_len is not None:
            assert self.max_seq_len % (self.mem_freq + 1) == 0
        self.model.set_mem_cache_args(mem_freq, top_k, max_cache_size)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        total_len = input_ids.shape[1]
        if past_key_values:
            prev_len = input_ids.shape[1] - 1
        else:
            prev_len = 0

        position_ids = kwargs.get("position_ids", None)

        if self.mem_freq is not None:
            if position_ids is not None:
                raise NotImplementedError
            # T = input_ids.shape[1]

            prev_incomplete_len = prev_len % self.mem_freq
            prev_complete_len = prev_len - prev_incomplete_len
            incomplete_len = total_len % self.mem_freq
            new_full_len = total_len - prev_complete_len - incomplete_len

            prev_input, input_ids_with_mem, input_ids_without_mem = torch.split(
                input_ids, (prev_complete_len, new_full_len, incomplete_len), dim=-1
            )

            bsz, _ = input_ids.size()
            input_ids_with_mem = input_ids_with_mem.view(bsz, -1, self.mem_freq)
            input_ids_with_mem = torch.cat(
                (
                    input_ids_with_mem,
                    input_ids_with_mem.new_full(
                        (bsz, input_ids_with_mem.shape[1], 1), self.mem_id
                    ),
                ),
                dim=-1,
            ).view(bsz, -1)
            input_ids = torch.cat(
                (prev_input, input_ids_with_mem, input_ids_without_mem), dim=-1
            )
            if attention_mask is not None:
                attention_mask_with_mem, attention_mask_without_mem = torch.split(
                    attention_mask,
                    (prev_complete_len + new_full_len, incomplete_len),
                    dim=-1,
                )
                attention_mask_with_mem = attention_mask_with_mem.view(
                    bsz, -1, self.mem_freq
                )
                attention_mask_with_mem = torch.cat(
                    (
                        attention_mask_with_mem,
                        attention_mask_with_mem.new_ones(
                            (bsz, attention_mask_with_mem.shape[1], 1)
                        ),
                    ),
                    dim=-1,
                ).view(bsz, -1)
                attention_mask = torch.cat(
                    (attention_mask_with_mem, attention_mask_without_mem), dim=-1
                )

        input_ids = input_ids[:, prev_len:]
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_ids.shape[1] :].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if (
            inputs_embeds is not None
            and past_key_values is None
            and self.mem_freq is None
        ):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "offload_cache_to_cpu": kwargs.get("offload_cache_to_cpu"),
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


def add_mem_tokens(example, mem_freq, mem_id):
    ids = example["input_ids"]
    ret = []
    prev_idx = 0
    for t_idx in range(mem_freq, len(ids), mem_freq):
        ret.extend(ids[prev_idx:t_idx])
        ret.append(mem_id)
        prev_idx = t_idx
    ret.extend(ids[prev_idx:])
    # drop attention_mask
    return {"input_ids": ret}


def patch_llama_with_landmark_attn():
    import transformers

    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
    transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb


def set_model_mem_id(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    mem_id = tokenizer.convert_tokens_to_ids(MEM_TOKEN)
    model.set_mem_id(mem_id)


def get_mem_id(tokenizer: LlamaTokenizer):
    return tokenizer.convert_tokens_to_ids(MEM_TOKEN)
