"""
Gemma 4 fused attention monkeypatch.

Replaces the per-layer RMSNorm + RoPE + transpose sequence with fused Triton
kernels, eliminating intermediate tensor allocations from rotate_half / apply_rotary_pos_emb

Usage:
    from axolotl.monkeypatch.models.gemma4.fused_attn import patch_gemma4_fused_attn
    patch_gemma4_fused_attn()
"""

import logging
import threading
from typing import Callable

import torch

logger = logging.getLogger(__name__)


# Thread-local side channel for the shared-KV dict. We route the dict through
# this instead of a kwarg on the decoder-layer forward so that when HF's
# GradientCheckpointingLayer forms `partial(super().__call__, **kwargs)` and
# axolotl's CPU_Offloaded_Gradient_Checkpointer captures that partial in
# `ctx.forward_function`, the ctx does not hold a reference to the dict --
# otherwise the K/V tensors stored in the dict for the producer layers stay
# pinned across the full backward pass (and, via Python ref-cycle delays in
# torch's caching allocator, bleed across training steps), causing VRAM to
# climb ~0.47 GiB/step under the hybrid FA2+SDPA path.
_GEMMA4_SHARED_KV_TLS = threading.local()


def _set_shared_kv_states(store):
    _GEMMA4_SHARED_KV_TLS.store = store


def _get_shared_kv_states():
    return getattr(_GEMMA4_SHARED_KV_TLS, "store", None)


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
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma4.modeling_gemma4 import (
            eager_attention_forward,
        )

        # Prefer the thread-local store (populated by the patched decoder-layer
        # __call__) so the dict is not captured by the checkpoint partial.
        tls_store = _get_shared_kv_states()
        if tls_store is not None:
            shared_kv_states = tls_store

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


def _patch_decoder_layer_call():
    """Strip `shared_kv_states` from the decoder-layer kwargs and route it via
    thread-local storage instead. This breaks the capture chain
    `ctx.forward_function -> partial -> kwargs -> shared_kv_states dict` inside
    the CPU-offload activation checkpointer, so the dict (and the K/V tensors
    it holds for the producer layers) is not pinned for the duration of the
    backward pass.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    if getattr(Gemma4TextDecoderLayer, "_axolotl_shared_kv_tls_patched", False):
        return

    original_call = Gemma4TextDecoderLayer.__call__

    def patched_call(self, *args, **kwargs):
        shared_kv = kwargs.pop("shared_kv_states", None)
        # Overwrite TLS unconditionally (including with None) so a previous
        # step's dict cannot leak into a later call that doesn't pass
        # `shared_kv_states`. The fallback branch in fused_forward relies on
        # TLS being None to defer to the kwarg path.
        _set_shared_kv_states(shared_kv)
        return original_call(self, *args, **kwargs)

    Gemma4TextDecoderLayer.__call__ = patched_call
    Gemma4TextDecoderLayer._axolotl_shared_kv_tls_patched = True


def patch_gemma4_fused_attn():
    """
    Monkeypatch Gemma4TextAttention.forward to use fused RMSNorm+RoPE kernels,
    and route `shared_kv_states` via thread-local storage to avoid a VRAM leak
    under activation checkpointing.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    original_forward = Gemma4TextAttention.forward
    Gemma4TextAttention.forward = _make_fused_forward(original_forward)

    _patch_decoder_layer_call()

    logger.info(
        "Patched Gemma4TextAttention.forward with fused RMSNorm+RoPE Triton kernels"
    )
