"""
Gemma 4 fused attention monkeypatch.

Replaces the per-layer RMSNorm + RoPE + transpose sequence with fused Triton
kernels, eliminating intermediate tensor allocations from rotate_half / apply_rotary_pos_emb

Usage:
    from axolotl.monkeypatch.models.gemma4.fused_attn import patch_gemma4_fused_attn
    patch_gemma4_fused_attn()
"""

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def _make_fused_forward(original_forward):
    """Create a patched forward that uses fused RMSNorm+RoPE kernels."""

    from axolotl.kernels.gemma4_fused_rope import (
        fused_rms_norm_noscale,
        fused_rms_norm_rope,
    )

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma4.modeling_gemma4 import (
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        eps = self.config.rms_norm_eps

        cos, sin = position_embeddings

        # ---- Projections ----
        # Use apply_qkv if present (LoRA kernel patch), otherwise direct proj
        has_lora_qkv = hasattr(self, "apply_qkv")

        if has_lora_qkv:
            query_states, key_states, value_states = self.apply_qkv(hidden_states)
            query_states = query_states.view(hidden_shape)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)

        # ---- Q path: fused q_norm + RoPE ----
        query_states = fused_rms_norm_rope(
            query_states,
            self.q_norm.weight,
            cos,
            sin,
            eps=eps,
        )
        query_states = query_states.transpose(1, 2)

        # ---- K/V path ----
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            if has_lora_qkv:
                # apply_qkv already computed k/v projections
                key_states = key_states.view(hidden_shape)
                value_states = (
                    value_states.view(hidden_shape)
                    if self.v_proj is not None
                    else key_states
                )
            else:
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                value_states = (
                    self.v_proj(hidden_states).view(hidden_shape)
                    if self.v_proj is not None
                    else key_states
                )

            # Fused k_norm + RoPE
            key_states = fused_rms_norm_rope(
                key_states,
                self.k_norm.weight,
                cos,
                sin,
                eps=eps,
            )
            key_states = key_states.transpose(1, 2)

            # Fused v_norm (no scale, no RoPE)
            value_states = fused_rms_norm_noscale(value_states, eps=eps)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = key_states, value_states

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return fused_forward


def patch_gemma4_fused_attn():
    """
    Monkeypatch Gemma4TextAttention.forward to use fused RMSNorm+RoPE kernels.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    original_forward = Gemma4TextAttention.forward
    Gemma4TextAttention.forward = _make_fused_forward(original_forward)

    logger.info(
        "Patched Gemma4TextAttention.forward with fused RMSNorm+RoPE Triton kernels"
    )
