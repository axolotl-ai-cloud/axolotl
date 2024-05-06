"""Flash attention monkey patch for llama model"""

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OriginalLlamaDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    apply_rotary_pos_emb,
    repeat_kv,
)
from xformers.ops import SwiGLU

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids, set_module_name

try:
    from flash_attn.flash_attn_interface import (  # pylint: disable=ungrouped-imports
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_kvpacked_func as flash_attn_varlen_kvpacked_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )


LOG = logging.getLogger("axolotl")


def is_xformers_swiglu_available() -> bool:
    from xformers.ops.common import get_xformers_operator

    try:
        get_xformers_operator("swiglu_packedw")()
        return True
    except RuntimeError as exc:
        if "No such operator xformers::swiglu_packedw " in str(exc):
            return False
        return True


def replace_llama_mlp_with_swiglu(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            mlp = FusedMLP(
                module.config, module.gate_proj, module.up_proj, module.down_proj
            )
            set_module_name(model, name, mlp)


def replace_llama_qkv_with_fused(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            qkv = FusedAttention(
                module.config,
                module.q_proj,
                module.k_proj,
                module.v_proj,
                module.o_proj,
            )
            set_module_name(model, name, qkv)


def replace_llama_attn_with_flash_attn(
    packed: Optional[bool] = False,
    cross_entropy: Optional[bool] = False,
    rms_norm: Optional[bool] = False,
    use_shifted_sparse_attn: Optional[bool] = False,
):
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (  # pylint: disable=protected-access
        _prepare_decoder_attention_mask
    )
    if use_shifted_sparse_attn:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            flashattn_forward_with_s2attn
        )
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            flashattn_forward
        )

    if packed:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
        transformers.models.llama.modeling_llama.LlamaModel.forward = (
            llama_model_forward
        )

    # skip only if explicitly disabled
    if cross_entropy:
        try:
            from flash_attn.losses.cross_entropy import CrossEntropyLoss

            LOG.info("patching with flash_attn.losses.cross_entropy")
            transformers.models.llama.modeling_llama.CrossEntropyLoss = partial(
                CrossEntropyLoss, inplace_backward=True
            )
        except ImportError:
            LOG.info(
                "optimized flash-attention CrossEntropyLoss not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=xentropy_cuda_lib&subdirectory=csrc/xentropy'`)"
            )

    # skip only if explicitly disabled
    if rms_norm:
        try:
            from flash_attn.ops.rms_norm import RMSNorm

            class LlamaRMSNorm(RMSNorm):
                """Patched LLamaRMSNorm"""

                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__(hidden_size, eps=eps)

            LOG.info("patching with flash_attn.ops.rms_norm")
            transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        except ImportError:
            LOG.info(
                "optimized flash-attention RMSNorm not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=dropout_layer_norm&subdirectory=csrc/layer_norm'`)"
            )


class FusedAttention(LlamaAttention):
    """
    Fused QKV Attention layer for incrementally improved training efficiency
    """

    def __init__(
        self,
        config,
        q: torch.nn.Linear,  # pylint: disable=invalid-name
        k: torch.nn.Linear,  # pylint: disable=invalid-name
        v: torch.nn.Linear,  # pylint: disable=invalid-name
        o: torch.nn.Linear,  # pylint: disable=invalid-name
    ):
        super().__init__(config)
        self.config = config
        self.init_device = next(iter(q.state_dict().values())).device

        # define equivalent fused qkv projection
        self.out_features: List[int] = [q.out_features, k.out_features, v.out_features]
        self.qkv_proj = torch.nn.Linear(
            q.in_features, sum(self.out_features), device=self.init_device, bias=False
        )
        self.o_proj = o

        # overwrite initialized weights with pretrained weights
        self.qkv_proj.weight.data = torch.cat(
            (q.weight.data, k.weight.data, v.weight.data), dim=0
        )

    def _post_training(self, model, name):
        q_proj, k_proj, v_proj = torch.split(
            self.qkv_proj.weight.data, self.out_features, dim=0
        )

        new_attn = LlamaAttention(self.config)
        new_attn.q_proj.weight.data = q_proj
        new_attn.k_proj.weight.data = k_proj
        new_attn.v_proj.weight.data = v_proj
        new_attn.o_proj.weight.data = self.o_proj.weight.data

        set_module_name(model, name, new_attn)


class FusedMLP(torch.nn.Module):
    """
    Fused MLP layer for incrementally improved training efficiency
    """

    def __init__(
        self,
        config,
        gate_proj: torch.nn.Linear,
        up_proj: torch.nn.Linear,
        down_proj: torch.nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.swiglu = SwiGLU(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=False,
            _pack_weights=True,
        )
        # overwrite initialized weights with pretrained weights
        self.swiglu.w12.weight.data = torch.cat(
            (gate_proj.weight.data, up_proj.weight.data), dim=0
        )
        self.swiglu.w3.weight.data = down_proj.weight.data

    def _post_training(self, model, name):
        w1, w2 = torch.split(  # pylint: disable=invalid-name
            self.swiglu.w12.weight.data, self.config.intermediate_size, dim=0
        )

        # Assign the split weights back to the original layers
        new_mlp = LlamaMLP(self.config)
        new_mlp.gate_proj.weight.data = w1
        new_mlp.up_proj.weight.data = w2
        new_mlp.down_proj.weight.data = self.swiglu.w3.weight.data

        set_module_name(model, name, new_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        return self.swiglu(x)


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


GROUP_SIZE_RATIO = 1 / 4


def flashattn_forward_with_s2attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,  # pylint: disable=unused-argument
    cu_seqlens: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    max_seqlen: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    From: https://github.com/dvlab-research/LongLoRA/blob/main/llama_attn_replace.py

    attention_mask: [bsz, q_len]

    `cu_seqlens` will be ignored if provided
    `max_seqlen` will be ignored if provided
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]
    # pylint: disable=duplicate-code

    cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]
    # shift

    group_size = int(q_len * GROUP_SIZE_RATIO)
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )

    qkv = (
        qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(bsz * 2, q_len, 3, self.num_heads // 2, self.head_dim)
    )
    x = rearrange(  # pylint: disable=invalid-name
        qkv, "b s three h d -> b s (three h d)"
    )
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    cu_q_len_tmp = torch.arange(
        0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype
    )
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + group_size // 2]).repeat(
        bsz, 1
    ) + cu_q_lens[:-1].unsqueeze(-1)
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)

    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    output = (
        output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim)
        .transpose(1, 2)
        .reshape(bsz, q_len, nheads, self.head_dim)
    )
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def flashattn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,  # pylint: disable=unused-argument
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
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
        if isinstance(self, FusedAttention):
            query_states, key_states, value_states = self.qkv_proj(hidden_states).split(
                self.out_features, dim=-1
            )
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

    cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
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
    # flash-attn v2 start
    #

    if self.training:
        # during training q,k,v always have same seqlen
        assert key_states.shape == query_states.shape
        is_causal = True
    else:
        # turn off FA causal mask after first inference autoregressive iteration
        # only on first autoregressive step q,k,v have same seqlen
        is_causal = key_states.shape == query_states.shape

    dropout_rate = 0.0 if not self.training else getattr(self, "attention_dropout", 0.0)

    if cu_seqlens is not None and max_seqlen is not None and cu_seqlens.dim() == 1:
        # special handling using sample packing
        qkv = torch.stack(
            [query_states, key_states, value_states], dim=2
        )  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
        qkv = rearrange(qkv, "b s ... -> (b s) ...")

        output = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=True,
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    elif query_states.shape == key_states.shape:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        qkv_unpad, cu_seqlens_q, max_seqlen_q, _, output_pad_fn = generate_qkv(
            query_states,
            key_states,
            value_states,
            qkvpacked=True,
            # We have disabled _prepare_decoder_attention_mask in LlamaModel
            # the attention_mask should be the same as the key_padding_mask
            key_padding_mask=attention_mask,
            query_padding_mask=attention_mask[:, -query_states.size(1) :]
            if attention_mask is not None
            else None,
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens_q,
            max_seqlen_q,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=is_causal,
        )
        output = output_pad_fn(output_unpad)
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if attention_mask is None or attention_mask.all().item():
            output = flash_attn_kvpacked_func(
                query_states,
                torch.stack([key_states, value_states], 2),
                dropout_p=dropout_rate,
                causal=is_causal,
            )
        else:
            (  # pylint: disable=unbalanced-tuple-unpacking
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query_states,
                key_states,
                value_states,
                kvpacked=True,
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query_states.size(1) :]
                if attention_mask is not None
                else None,
            )
            if q_unpad.dtype != kv_unpad.dtype:
                kv_unpad = kv_unpad.to(q_unpad.dtype)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_rate,
                softmax_scale=None,
                causal=is_causal,
            )
            output = output_pad_fn(output_unpad)

    attn_output = output
    if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

    #
    # flash-attn v2 end
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


# based on https://github.com/Dao-AILab/flash-attention/blob/364a5b/tests/test_flash_attn.py#L38
def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    kvpacked=False,
    qkvpacked=False,
):  # pylint: disable=invalid-name,unnecessary-lambda-assignment
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            q, query_padding_mask
        )

        output_pad_fn = lambda output_unpad: pad_input(  # noqa: E731
            output_unpad, indices_q, batch_size, seqlen_q
        )

    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q_unpad.device,
        )
        max_seqlen_q = seqlen_q

        output_pad_fn = lambda output_unpad: rearrange(  # noqa: E731
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=k_unpad.device,
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        return (qkv_unpad, cu_seqlens_q, max_seqlen_q, qkv, output_pad_fn)

    if kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        return (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
        )

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
    )


def llama_model_forward(
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
    cache_position: Optional[  # pylint: disable=unused-argument
        torch.LongTensor
    ] = None,
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
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    cu_seqlens = None
    max_seqlen = None
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
        cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)
        cu_seqlens = cu_seqlens.squeeze()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = (
        self._prepare_decoder_attention_mask(  # pylint: disable=protected-access
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            transformers.logger.warning_once(
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

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(
                        *inputs,
                    )

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                None,
                padding_mask,
                cu_seqlens,
                max_seqlen,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
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


class LlamaDecoderLayer(OriginalLlamaDecoderLayer):
    """
    patched version of LlamaDecoderLayer to pass through the precalculated cu_seqlens
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
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
            cu_seqlens (`torch.Tensor`, *optional*) cumulative sequence len when packing
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
            padding_mask=padding_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
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
