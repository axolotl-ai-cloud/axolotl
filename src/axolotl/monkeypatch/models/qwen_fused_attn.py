"""Qwen fused q/k RMSNorm + RoPE attention monkeypatches."""

from __future__ import annotations

from typing import Callable

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _attention_interface(
    functions, implementation: str, fallback: Callable
) -> Callable:
    if hasattr(functions, "get_interface"):
        return functions.get_interface(implementation, fallback)
    if implementation == "eager":
        return fallback
    return functions[implementation]


def _norm_weight_eps(norm) -> tuple[torch.Tensor, float]:
    if hasattr(norm, "variance_epsilon"):
        return norm.weight, norm.variance_epsilon
    if hasattr(norm, "eps"):
        return 1.0 + norm.weight, norm.eps
    return norm.weight, 1e-6


def _make_qwen_forward(modeling, *, gated_q: bool, include_sliding_window: bool):
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if hasattr(self, "apply_qkv"):
            query_raw, key_raw, value_raw = self.apply_qkv(hidden_states)
        else:
            query_raw = self.q_proj(hidden_states)
            key_raw = self.k_proj(hidden_states)
            value_raw = self.v_proj(hidden_states)

        gate = None
        if gated_q:
            query_states, gate = torch.chunk(
                query_raw.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
        else:
            query_states = query_raw.view(hidden_shape)

        key_states = key_raw.view(hidden_shape)
        value_states = value_raw.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_weight, q_eps = _norm_weight_eps(self.q_norm)
        k_weight, k_eps = _norm_weight_eps(self.k_norm)
        query_states = fused_rms_norm_rope(
            query_states,
            q_weight,
            cos,
            sin,
            eps=q_eps,
        ).transpose(1, 2)
        key_states = fused_rms_norm_rope(
            key_states,
            k_weight,
            cos,
            sin,
            eps=k_eps,
        ).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = _attention_interface(
            modeling.ALL_ATTENTION_FUNCTIONS,
            self.config._attn_implementation,
            modeling.eager_attention_forward,
        )

        attn_kwargs = {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            **kwargs,
        }
        if include_sliding_window:
            attn_kwargs["sliding_window"] = self.sliding_window

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **attn_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        if hasattr(self, "apply_o"):
            attn_output = self.apply_o(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return fused_forward


def _patch_attention_class(attn_cls, forward, label: str) -> bool:
    if getattr(attn_cls, "_axolotl_fused_qk_norm_rope", False):
        return False
    attn_cls.forward = forward
    attn_cls._axolotl_fused_qk_norm_rope = True
    LOG.info("Patched %s.forward with fused q/k RMSNorm+RoPE Triton kernels", label)
    return True


def patch_qwen3_fused_attn() -> bool:
    from transformers.models.qwen3 import modeling_qwen3

    return _patch_attention_class(
        modeling_qwen3.Qwen3Attention,
        _make_qwen_forward(
            modeling_qwen3,
            gated_q=False,
            include_sliding_window=True,
        ),
        "Qwen3Attention",
    )


def patch_qwen3_5_fused_attn() -> bool:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    return _patch_attention_class(
        modeling_qwen3_5.Qwen3_5Attention,
        _make_qwen_forward(
            modeling_qwen3_5,
            gated_q=True,
            include_sliding_window=False,
        ),
        "Qwen3_5Attention",
    )


def patch_qwen3_5_moe_fused_attn() -> bool:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    return _patch_attention_class(
        modeling_qwen3_5_moe.Qwen3_5MoeAttention,
        _make_qwen_forward(
            modeling_qwen3_5_moe,
            gated_q=True,
            include_sliding_window=False,
        ),
        "Qwen3_5MoeAttention",
    )


def patch_qwen_fused_attn(model_config_type: str) -> bool:
    if model_config_type == "qwen3":
        return patch_qwen3_fused_attn()
    if model_config_type == "qwen3_5":
        return patch_qwen3_5_fused_attn()
    if model_config_type == "qwen3_5_moe":
        return patch_qwen3_5_moe_fused_attn()
    LOG.warning(
        "nvfp4_training.fuse_qk_norm_rope is only wired for qwen3, qwen3_5, "
        "and qwen3_5_moe; got model_config_type=%s",
        model_config_type,
    )
    return False
