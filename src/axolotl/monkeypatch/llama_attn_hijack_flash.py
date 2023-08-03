"""Flash attention monkey patch for llama model"""

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

from typing import Optional, Tuple

import torch
import transformers
from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def get_cu_seqlens(attn_mask):
    device = attn_mask.device
    # Exclude zeros to avoid adding their positions to the mask
    t_non_zeros = attn_mask[attn_mask != 0]
    # Find where the sequence number changes (including the first position)
    seq_change = torch.cat(
        [
            torch.tensor([1], dtype=torch.int32, device=device),
            t_non_zeros[1:] != t_non_zeros[:-1],
        ]
    )
    # Get the indices where the sequence changes
    change_indices = torch.cat(
        [
            (seq_change == 1).nonzero(as_tuple=True)[0],
            torch.tensor([len(t_non_zeros)], dtype=torch.int32, device=device),
        ]
    )
    # Calculate the sequence lengths
    seq_lengths = change_indices[1:] - change_indices[:-1]
    # Calculate the cumulative sequence lengths
    cu_seqlens = torch.cat(
        [torch.tensor([0], dtype=torch.int32, device=device), seq_lengths.cumsum(0)]
    )

    max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    return cu_seqlens.to(dtype=torch.int32), max_seq_len


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    # pylint: disable=duplicate-code
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
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0,
            (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=qkv.device,
        )
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    else:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        cu_q_lens, max_s = get_cu_seqlens(key_padding_mask)

        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)

    return (
        self.o_proj(rearrange(output, "b s h d -> b s (h d)")),
        None,
        None,
    )


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
):  # pylint: disable=unused-argument
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (  # pylint: disable=protected-access
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
