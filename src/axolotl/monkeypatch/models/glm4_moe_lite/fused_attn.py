"""Fuse partial rotary embeddings and MLA projection assembly."""

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _make_fused_forward(original_forward):
    from axolotl.kernels.glm4_moe_lite import fused_mla_prepare

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        cos, sin = position_embeddings
        if (
            not hidden_states.is_cuda
            or hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or past_key_values is not None
            or cos.requires_grad
            or sin.requires_grad
        ):
            return original_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            eager_attention_forward,
        )

        b, s = hidden_states.shape[:2]
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_rot = compressed.split(
            (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(latent))
        if any(x.dtype != q.dtype for x in (kv, k_rot, cos, sin)):
            return original_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        q, k, v = fused_mla_prepare(
            q.view(b, s, self.num_heads, self.qk_head_dim),
            kv.view(b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
            k_rot,
            cos,
            sin,
            self.qk_nope_head_dim,
            self.config.rope_interleave,
        )
        attention = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention(
            self,
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        return self.o_proj(output.reshape(b, s, -1).contiguous()), weights

    return forward


def patch_glm4_moe_lite_fused_attn():
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention,
    )

    if getattr(Glm4MoeLiteAttention, "_axolotl_fused_attn_patched", False):
        return
    Glm4MoeLiteAttention.forward = _make_fused_forward(Glm4MoeLiteAttention.forward)
    Glm4MoeLiteAttention._axolotl_fused_attn_patched = True
    LOG.info(
        "Patched GLM-4 MoE Lite attention with fused partial RoPE and MLA Q/K/V assembly"
    )
