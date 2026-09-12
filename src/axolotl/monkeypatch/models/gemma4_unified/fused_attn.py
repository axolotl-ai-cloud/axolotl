"""
Gemma 4 Unified fused attention monkeypatch.

Mirrors :mod:`axolotl.monkeypatch.models.gemma4.fused_attn` for the
encoder-free ``gemma4_unified`` text backbone (``Gemma4UnifiedTextAttention``),
replacing the per-layer RMSNorm + RoPE + transpose sequence with fused Triton
kernels.

The math is identical to standard Gemma 4 (q_norm/k_norm with scale + RoPE,
v_norm without scale, no RoPE), so the same kernels are reused. Like standard
Gemma 4 on transformers 5.10.x, the unified attention keys ``shared_kv_states``
by **layer type string** (``self.layer_type``); the separate module is needed
only because the unified backbone redefines its classes in its own namespace
(it does not modular-import from ``modeling_gemma4``).

Usage:
    from axolotl.monkeypatch.models.gemma4_unified.fused_attn import (
        patch_gemma4_unified_fused_attn,
    )
    # Pass install_shared_kv_workaround=True when activation checkpointing is enabled.
    patch_gemma4_unified_fused_attn(install_shared_kv_workaround=True)
"""

from typing import Callable

import torch

from axolotl.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level dict used as a side channel for shared KV states avoiding kwarg
# and TLS to prevent a memory leak under gradient checkpointing (PR #3611).
# Kept separate from the standard-gemma4 store so the two never alias.
_GEMMA4_UNIFIED_SHARED_KV_STORE: dict = {"store": None}


def _set_shared_kv_states(store):
    _GEMMA4_UNIFIED_SHARED_KV_STORE["store"] = store


def _get_shared_kv_states():
    return _GEMMA4_UNIFIED_SHARED_KV_STORE["store"]


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
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            eager_attention_forward,
        )

        store = _get_shared_kv_states()
        if store is not None:
            shared_kv_states = store

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        eps = self.config.rms_norm_eps

        cos, sin = position_embeddings

        # ---- Projections ----
        # Use apply_qkv if present (LoRA kernel patch), otherwise direct proj.
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
            # Unified keys the shared cache by layer-type string, not index.
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            if has_lora_qkv:
                # apply_qkv already computed k/v projections.
                key_states = key_states.view(hidden_shape)
                value_states = (
                    value_states.view(hidden_shape)
                    if self.v_proj is not None
                    else key_states
                )
            else:
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                # attention_k_eq_v: value reuses the (pre-norm) k_proj output.
                value_states = (
                    self.v_proj(hidden_states).view(hidden_shape)
                    if self.v_proj is not None
                    else key_states
                )

            # Fused k_norm + RoPE (creates a new tensor, leaving value_states
            # pointing at the pre-norm k_proj output for the k_eq_v case).
            key_states = fused_rms_norm_rope(
                key_states,
                self.k_norm.weight,
                cos,
                sin,
                eps=eps,
            )
            key_states = key_states.transpose(1, 2)

            # Fused v_norm (no scale, no RoPE).
            value_states = fused_rms_norm_noscale(value_states, eps=eps)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key_states, value_states

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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
        # Use apply_o if present (LoRA O kernel patch), otherwise direct proj.
        if hasattr(self, "apply_o"):
            attn_output = self.apply_o(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return fused_forward


def _patch_decoder_layer_call():
    """Strip ``shared_kv_states`` from decoder-layer kwargs and route via the
    module-level side channel so the checkpoint partial cannot pin it (PR #3611).
    """
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextDecoderLayer,
    )

    if getattr(Gemma4UnifiedTextDecoderLayer, "_axolotl_shared_kv_patched", False):
        return

    original_call = Gemma4UnifiedTextDecoderLayer.__call__

    def patched_call(self, *args, **kwargs):
        shared_kv = kwargs.pop("shared_kv_states", None)
        # Overwrite unconditionally (including with None) so a previous step's
        # dict cannot leak into a later call without shared_kv_states (PR #3611).
        _set_shared_kv_states(shared_kv)
        return original_call(self, *args, **kwargs)

    Gemma4UnifiedTextDecoderLayer.__call__ = patched_call
    Gemma4UnifiedTextDecoderLayer._axolotl_shared_kv_patched = True


def patch_gemma4_unified_fused_attn(install_shared_kv_workaround: bool = False):
    """
    Monkeypatch ``Gemma4UnifiedTextAttention.forward`` to use fused RMSNorm+RoPE
    kernels, and optionally route ``shared_kv_states`` via a module-level side
    channel to avoid a VRAM leak under activation checkpointing (PR #3611).
    """
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextAttention,
    )

    if getattr(Gemma4UnifiedTextAttention, "_axolotl_fused_attn_patched", False):
        return

    original_forward = Gemma4UnifiedTextAttention.forward
    Gemma4UnifiedTextAttention.forward = _make_fused_forward(original_forward)
    Gemma4UnifiedTextAttention._axolotl_fused_attn_patched = True

    if install_shared_kv_workaround:
        _patch_decoder_layer_call()

    logger.info(
        "Patched Gemma4UnifiedTextAttention.forward with fused RMSNorm+RoPE "
        "Triton kernels"
    )
    if install_shared_kv_workaround:
        logger.info("Installed Gemma4Unified shared_kv_states side channel (PR #3611)")
