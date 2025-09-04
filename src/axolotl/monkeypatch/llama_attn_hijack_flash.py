"""Flash attention monkey patch for llama model"""

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

import importlib.util
import warnings
from typing import Optional, Tuple

import torch
import transformers
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    apply_rotary_pos_emb,
    repeat_kv,
)

from axolotl.monkeypatch.utils import set_module_name
from axolotl.utils.logging import get_logger

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )


LOG = get_logger(__name__)


def is_xformers_available() -> bool:
    return importlib.util.find_spec("xformers") is not None


def is_xformers_swiglu_available() -> bool:
    if not is_xformers_available():
        return False

    from xformers.ops.common import get_xformers_operator

    try:
        get_xformers_operator("swiglu_packedw")()
        return True
    except RuntimeError as exc:
        if "No such operator xformers::swiglu_packedw " in str(exc):
            return False
        return True


def replace_llama_mlp_with_swiglu(model):
    if is_xformers_swiglu_available():
        from axolotl.monkeypatch.xformers_ import FusedMLP
    else:
        raise RuntimeError("xformers SwiGLU not available for this environment")

    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            mlp = FusedMLP(
                module.config, module.gate_proj, module.up_proj, module.down_proj
            )
            set_module_name(model, name, mlp)


def patch_fa_llama_cross_entropy():
    LOG.info(
        "patching transformers.loss.loss_utils.fixed_cross_entropy with flash_attn.ops.triton.cross_entropy"
    )
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_attn_cross_entropy_loss,
    )

    def fa2_fixed_cross_entropy(
        source,
        target,
        num_items_in_batch: int = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss, _ = flash_attn_cross_entropy_loss(
            source, target, ignore_index=ignore_index
        )
        if reduction == "sum":
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.sum() / (target != ignore_index).sum()
        return loss

    transformers.loss.loss_utils.fixed_cross_entropy = fa2_fixed_cross_entropy


def patch_llama_rms_norm():
    try:
        from flash_attn.ops.rms_norm import RMSNorm

        class LlamaRMSNorm(RMSNorm):
            """Patched LLamaRMSNorm"""

            def __init__(self, hidden_size, eps=1e-6):
                super().__init__(hidden_size, eps=eps)

        LOG.info("patching with flash_attn.ops.rms_norm")
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
    except ImportError:
        LOG.warning(
            "optimized flash-attention RMSNorm not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=dropout_layer_norm&subdirectory=csrc/layer_norm'`)"
        )


def replace_llama_attn_with_flash_attn(
    cross_entropy: Optional[bool] = False,
    rms_norm: Optional[bool] = False,
    use_shifted_sparse_attn: Optional[bool] = False,
):
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    if use_shifted_sparse_attn:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            flashattn_forward_with_s2attn
        )

    # skip only if explicitly disabled
    if cross_entropy:
        patch_fa_llama_cross_entropy()

    # skip only if explicitly disabled
    if rms_norm:
        patch_llama_rms_norm()


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
):
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
    padding_mask: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    From: https://github.com/dvlab-research/LongLoRA/blob/main/llama_attn_replace.py

    attention_mask: [bsz, q_len]

    `cu_seqlens` will be ignored if provided
    `max_seqlen` will be ignored if provided
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.",
            stacklevel=2,
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
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
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
