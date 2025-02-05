"""
LoLCATs attention combining sliding window and linear attentions
- Using the TK "terracing" arrangement
- Training over long sequences with fixed memory with recurrent view
- During attention transfer, use Flash Attention to compute softmax attention outputs

For each layer:
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

try:
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
except ModuleNotFoundError:
    _flash_attention_forward = None  # Transformers v4.36

from .linear_attention import softmax_attention
from .linear_window_attention_tk import LolcatsTKWindowAttention
from .rotary import apply_rotary_pos_emb

LOG = logging.getLogger(
    "axolotl.integrations.lolcats.linear_attention.linear_window_attention_tk_long"
)


class LolcatsTKWindowLongAttention(LolcatsTKWindowAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """

    def __init__(self, remove_base_attn=True, **kwargs):
        # keep self.base_attn for Flash Attention inference
        super().__init__(remove_base_attn=True, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        if self.train_attention and self.base_inference:
            with torch.no_grad():
                # LOG.debug(hidden_states.shape)
                _y_true = flash_attention_2(
                    self,  # self.base_attn,
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    # output_hidden_states=False,
                    use_cache=False,
                )[0]
                # _y_true.shape is (batch_size, seq_len, num_heads, head_dim)
                y_true = _y_true.reshape(b, l, -1).contiguous()
                y_true = self.o_proj(y_true)
                layer_io = (hidden_states, _y_true)  # hack
                # layer_io = (hidden_states.cpu(), _y_true.cpu())  # hack
                return y_true, layer_io, None

        q, k, v, kv_seq_len = self.process_qkv(
            hidden_states, attention_mask, position_ids, past_key_value
        )
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)

        # attention_mask = None  # For now this is always True
        if past_key_value is None:  # Regular training
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred, a_pred = self.quadratic_attention(
                q,
                k,
                f_q,
                f_k,
                v,
                window_factors,
                linear_factors,
                window_size=self.window_size,
            )
        else:
            past_key_value.window_size = self.decode_window_size
            if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:  # Generating
                assert use_cache is True
                _kv = past_key_value.update_for_decoding(
                    k, v, self.layer_idx, self.feature_map_k, dtype=q.dtype
                )
                k_cache, v_cache, f_kv_state, f_k_state = _kv

                # Sliding window + linear attention decode
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = (
                    1 - window_factors if self.affine_attention_factors else 1
                )

                a_sm = torch.einsum("bhmd,bhnd->bhmn", q.float(), k_cache.float()) * (
                    k.shape[-1] ** -0.5
                )
                # a_sm = torch.softmax(a_sm, dim=-1)
                a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                a_sm = window_factors * torch.exp(a_sm - a_sm_max)
                sum_sm = a_sm.sum(dim=-1, keepdim=True)

                y_pred = torch.einsum(
                    "bhmn,bhnd->bhmd", a_sm, v_cache.float()
                ) + linear_factors * torch.einsum(
                    "bhlf,bhfd->bhld", f_q.float(), f_kv_state.float()
                )
                sum_ln = (
                    linear_factors
                    * torch.einsum("bhlf,bhnf->bhl", f_q.float(), f_k_state.float())[
                        ..., None
                    ]
                )
                y_pred = (y_pred / (sum_sm + sum_ln)).to(q.dtype)

            else:  # Stateful training
                if (
                    self.state_grad_enabled
                    and self.layer_idx == 0
                    and position_ids is not None
                ):
                    LOG.debug(
                        f"\n position_ids: [{position_ids[0, 0]}, {position_ids[0, -1]}]"
                    )
                    LOG.debug(
                        f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}"
                    )
                try:
                    kv_state = past_key_value.kv_states[self.layer_idx]
                    k_state = past_key_value.k_states[self.layer_idx]
                except IndexError:
                    kv_state, k_state = None, None
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = (
                    1 - window_factors if self.affine_attention_factors else 1
                )
                y_pred, a_pred = self.quadratic_attention(
                    q,
                    k,
                    f_q,
                    f_k,
                    v,
                    window_factors,
                    linear_factors,
                    window_size=self.window_size,
                    kv_state=kv_state,
                    k_state=k_state,
                )
                # Save and update KV cache and states
                # past_key_value.update(k, v.detach(), self.layer_idx,
                #                       fmap_key_states=f_k.detach(),
                #                       accumulate_in_fp32=True)
                past_key_value.update(
                    k, v, self.layer_idx, fmap_key_states=f_k, accumulate_in_fp32=True
                )

        # Concatenate heads and apply output projection
        _y_pred = y_pred.transpose(1, 2).contiguous()
        y_pred = self.o_proj(_y_pred.view(b, l, self.hidden_size))

        if self.train_attention:
            with torch.no_grad():
                a_true = softmax_attention(q, k, None, causal=True)[1]
            attn_weights = (_y_pred, (a_pred, a_true))
        else:
            attn_weights = _y_pred  # flash_attn outputs are shape (b, l, h, d)
        return y_pred, attn_weights, past_key_value


# -----------------
# Flash Attention 2
# -----------------


def flash_attention_2(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    """
    Wrapper for LlamaFlashAttention2
    Copied and modified from HF Transformers v4.36 and v4.43 implementations
    - (4.43) https://github.com/huggingface/transformers/blob/868d36d29ec132deeaaf8571b25b6a1b911d0145/src/transformers/models/llama/modeling_llama.py#L402
    - (4.36) https://github.com/huggingface/transformers/blob/a7cab3c283312b8d4de5df3bbe719971e24f4281/src/transformers/models/llama/modeling_llama.py#L456
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    try:  # As in Transformers v4.36
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    except Exception:  # As in Transformers v4.39
        cos, sin = self.rotary_emb(key_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        LOG.debug(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    if getattr(self, "_flash_attention_forward", False):
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            is_causal=True,
        )
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0,  # dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=True,
        )
    return attn_output, past_key_value
