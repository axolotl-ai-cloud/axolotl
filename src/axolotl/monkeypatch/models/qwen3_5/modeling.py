"""Monkeypatch for Qwen3_5 and Qwen3_5Moe models to pass position_ids to linear attention.

Qwen3-Next uses a different GatedDeltaNet interface (in_proj_qkvz / in_proj_ba /
fix_query_key_value_ordering) than Qwen3.5 / Qwen3.5-MoE (in_proj_qkv / in_proj_z /
in_proj_b / in_proj_a), so each family gets its own patched forward factory.
"""

import importlib
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from fla.modules.convolution import (
        causal_conv1d as fla_causal_conv1d,  # FLA >= 0.4.1
    )
except ImportError:
    try:
        from fla.modules.conv import causal_conv1d as fla_causal_conv1d  # FLA < 0.4.1
    except ImportError:
        fla_causal_conv1d = None


def get_cu_seqlens(position_ids):
    """
    Adapted from transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids.

    https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/modeling_flash_attention_utils.py#L316

    Handles 3-D MRoPE position_ids of shape [axes, batch, seq_len] (Qwen3.5).
    Qwen3.5 uses Multi-dimensional RoPE (MRoPE) where position_ids are expanded
    to [num_rope_axes, B, T] before being passed to decoder layers.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    All axes carry the same temporal positions for text-only finetuning, so we
    take axis 0 to recover the standard [B, T] layout for boundary detection.
    """
    if position_ids.ndim == 3:
        position_ids = position_ids[0]

    tensor_kwargs = {"dtype": torch.long, "device": position_ids.device}

    position_ids = position_ids.reshape(-1)
    indices_q = (position_ids == 0).nonzero().reshape(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )

    return cu_seq_lens_q


def _inject_fla_kernels(module) -> None:
    """Inject FLA kernels into a modeling module, bypassing is_flash_linear_attention_available."""
    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        module.FusedRMSNormGated = FusedRMSNormGated
        module.chunk_gated_delta_rule = chunk_gated_delta_rule
        module.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        module.is_fast_path_available = True
    except ImportError:
        module.chunk_gated_delta_rule = None
        module.fused_recurrent_gated_delta_rule = None
        module.FusedRMSNormGated = None


def _patched_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    """Shared decoder layer forward — threads position_ids into linear attention."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # MoE layers return (hidden_states, router_logits) — unpack
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states


def _make_qwen3_5_gated_delta_forward(apply_mask_fn):
    """
    Factory: returns the patched Qwen3_5 / Qwen3_5Moe GatedDeltaNet forward.
    Qwen3.5 uses four separate projections: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a.
    """

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = apply_mask_fn(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # Compute cu_seqlens for use by both causal_conv1d and chunk_gated_delta_rule
        cu_seqlens = None
        if not use_precomputed_states and position_ids is not None:
            cu_seqlens = get_cu_seqlens(position_ids=position_ids)

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        # Qwen3.5 uses four separate projections (unlike Qwen3-Next's combined ones)
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, D, T] for conv1d

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            # Inference single-token path: causal_conv1d_update expects [B, D, T]
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, D]
        else:
            if cache_params is not None:
                # Cache stores [B, D, T]
                cache_params.conv_states[self.layer_idx] = F.pad(
                    mixed_qkv,
                    (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                )

            if fla_causal_conv1d is not None:
                # FLA Triton kernel: expects [B, T, D], outputs [B, T, D]
                # cu_seqlens resets conv state at packed-sequence boundaries
                mixed_qkv, _ = fla_causal_conv1d(
                    x=mixed_qkv.transpose(1, 2),  # [B, T, D]
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cu_seqlens=cu_seqlens,
                )  # [B, T, D]
            else:
                # PyTorch fallback (no cu_seqlens support)
                if cu_seqlens is not None and cu_seqlens.shape[0] > batch_size + 1:
                    raise RuntimeError(
                        "Packed sequences require fla.modules.convolution.causal_conv1d "
                        "(cu_seqlens support). Install flash-linear-attention or disable packing."
                    )
                LOG.warning_once(
                    "FLA causal_conv1d not available. Falling back to PyTorch conv1d."
                )
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])  # [B, D, T]
                mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, D]

        # mixed_qkv is [B, T, D] in all paths from here
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g.to(dtype=query.dtype),
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g.to(dtype=query.dtype),
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)

    return patched_forward


# ---------------------------------------------------------------------------
# Unified patch entry point
# ---------------------------------------------------------------------------


def _apply_packing_patches(
    model_type: str,
    cls_prefix: str,
    forward_factory,
) -> None:
    """
    Apply all sample-packing patches for a qwen3_5 family model.

    Args:
        model_type:      transformers model_type string, e.g. "qwen3_5" or "qwen3_5_moe"
        cls_prefix:      class name prefix, e.g. "Qwen3_5" or "Qwen3_5Moe"
        forward_factory: factory that builds the patched GatedDeltaNet forward
    """
    module_name = f"transformers.models.{model_type}.modeling_{model_type}"

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        LOG.warning(f"{model_type} not found in transformers, skipping packing patches")
        return

    _inject_fla_kernels(module)

    getattr(module, f"{cls_prefix}DecoderLayer").forward = _patched_decoder_forward

    gated_cls = getattr(module, f"{cls_prefix}GatedDeltaNet")
    gated_cls.forward = forward_factory(module.apply_mask_to_padding_states)

    LOG.info(
        f"Applied {cls_prefix} packing patch (fla_causal_conv1d={'available' if fla_causal_conv1d else 'unavailable'})"
    )


def patch_qwen3_5_modeling_packing():
    _apply_packing_patches("qwen3_5", "Qwen3_5", _make_qwen3_5_gated_delta_forward)


def patch_qwen3_5_moe_modeling_packing():
    _apply_packing_patches(
        "qwen3_5_moe", "Qwen3_5Moe", _make_qwen3_5_gated_delta_forward
    )
