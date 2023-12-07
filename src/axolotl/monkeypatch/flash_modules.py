import torch
import logging
import warnings
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from typing import Optional, Tuple
from flash_attn.bert_padding import pad_input, unpad_input
from axolotl.monkeypatch.fused_modules import FusedAttention

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

def flashattn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
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

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = self.apply_rotary_fn(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    use_sliding_windows = (
        hasattr(self.config, "sliding_window") is not None
        and kv_seq_len > self.config.sliding_window
    )

    if use_sliding_windows:
        window_size = (self.config.sliding_window, self.config.sliding_window)
    else:
        window_size = (-1, -1)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        if (
            hasattr(self.config, "sliding_window")
            and kv_seq_len > self.config.sliding_window
        ):
            slicing_tokens = kv_seq_len - self.config.sliding_window

            past_key = past_key_value[0]
            past_value = past_key_value[1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key much have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            past_key_value = (past_key, past_value) if use_cache else None

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = self.repeat_kv_fn(key_states, self.num_key_value_groups)
    value_states = self.repeat_kv_fn(value_states, self.num_key_value_groups)

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
            window_size=window_size,
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
            window_size=window_size,
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
                window_size=window_size,
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
                window_size=window_size,
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

def replace_cross_entropy(modeling_class, module_name):
    """
    modeling_class: transformers.models.llama.modeling_<class>
    module_name: CrossEntropyLoss
    """
    try:
        from flash_attn.losses.cross_entropy import CrossEntropyLoss

        LOG.info("patching with flash_attn.losses.cross_entropy")

        cross_entropy_loss = partial(
            CrossEntropyLoss, inplace_backward=True
        )

        setattr(modeling_class, module_name, cross_entropy_loss)
        
    except ImportError:
        LOG.info(
            "optimized flash-attention CrossEntropyLoss not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=xentropy_cuda_lib&subdirectory=csrc/xentropy'`)"
        )

def replace_rms_norm(modeling_class, module_name):
    """
    modeling_class: transformers.models.llama.modeling_<class>
    module_name: RMSNorm
    """
    try:
        from flash_attn.ops.rms_norm import RMSNorm

        class FlashRMSNorm(RMSNorm):
            """A faster RMS Norm."""
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__(hidden_size, eps=eps)
        
        LOG.info("patching with flash_attn.ops.rms_norm")
        setattr(modeling_class, module_name, FlashRMSNorm)
    except ImportError:
        LOG.info(
            "optimized flash-attention RMSNorm not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=dropout_layer_norm&subdirectory=csrc/layer_norm'`)"
        )