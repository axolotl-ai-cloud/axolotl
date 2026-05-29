# Why: fuse Qwen3-VL q_norm/k_norm + mRoPE into one Triton kernel.

from typing import Callable

import torch

from axolotl.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_norm_module(norm):
    # Why: ModulesToSaveWrapper stores trainable norm weights per active adapter.
    modules_to_save = getattr(norm, "modules_to_save", None)
    if not modules_to_save:
        return norm
    adapters = getattr(norm, "active_adapters", None)
    if adapters is None:
        adapter = getattr(norm, "active_adapter", None)
        adapters = [adapter] if adapter is not None else []
    elif isinstance(adapters, str):
        adapters = [adapters]
    for name in adapters:
        if isinstance(name, str) and name in modules_to_save:
            return modules_to_save[name]
    return getattr(norm, "original_module", norm)


def _make_fused_forward():
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_norm = _resolve_norm_module(self.q_norm)
        k_norm = _resolve_norm_module(self.k_norm)
        eps = getattr(q_norm, "eps", None)
        if eps is None:
            eps = q_norm.variance_epsilon

        cos, sin = position_embeddings

        if hasattr(self, "apply_qkv"):
            query_states, key_states, value_states = self.apply_qkv(hidden_states)
            query_states = query_states.view(hidden_shape)
            key_states = key_states.view(hidden_shape)
            value_states = value_states.view(hidden_shape)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

        # accelerate's per-module pre-hooks that move CPU-staged params don't fire on a monkeypatched forward.
        q_w = q_norm.weight
        if q_w.device != query_states.device:
            q_w = q_w.to(query_states.device, non_blocking=True)
        k_w = k_norm.weight
        if k_w.device != key_states.device:
            k_w = k_w.to(key_states.device, non_blocking=True)

        query_states = fused_rms_norm_rope(
            query_states, q_w, cos, sin, eps=eps
        ).transpose(1, 2)
        key_states = fused_rms_norm_rope(key_states, k_w, cos, sin, eps=eps).transpose(
            1, 2
        )
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if hasattr(self, "apply_o"):
            attn_output = self.apply_o(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return fused_forward


def patch_qwen3_vl_fused_attn() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    if getattr(Qwen3VLTextAttention, "_axolotl_fused_attn_patched", False):
        return

    Qwen3VLTextAttention.forward = _make_fused_forward()
    Qwen3VLTextAttention._axolotl_fused_attn_patched = True
    logger.info(
        "Patched Qwen3VLTextAttention.forward with fused RMSNorm+mRoPE Triton kernel"
    )
