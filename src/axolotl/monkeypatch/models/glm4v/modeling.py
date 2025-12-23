"""
Monkeypatches for GLM4V models.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.glm4v.modeling_glm4v import (
    ALL_ATTENTION_FUNCTIONS,
    Glm4vTextAttention,
    Glm4vTextRotaryEmbedding,
    Glm4vTextConfig,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_glm4v_attention_rope_scaling():
    """
    Patch Glm4vTextAttention and Glm4vTextRotaryEmbedding to handle rope_parameters
    and partial rotary factor (for GLM-4.6V).
    """
    # Patch RotaryEmbedding __init__ to handle partial_rotary_factor and rope_parameters
    def patched_rotary_init(self, config: Glm4vTextConfig, device=None):
        nn.Module.__init__(self)

        rope_parameters = getattr(config, "rope_parameters", None)
        head_dim = config.hidden_size // config.num_attention_heads

        self.dim = head_dim
        base = config.rope_theta
        self.rope_type = "default"

        if rope_parameters is not None:
            self.rope_type = rope_parameters.get("rope_type", "default")
            factor = rope_parameters.get("partial_rotary_factor", 1.0)
            self.dim = int(head_dim * factor)
            base = rope_parameters.get("rope_theta", config.rope_theta)
        elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
            base = config.rope_theta

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # Initialize inv_freq manually
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 1.0

    Glm4vTextRotaryEmbedding.__init__ = patched_rotary_init

    # Patch Attention forward
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        mrope_section = None
        # Check rope_parameters first
        if hasattr(self.config, "rope_parameters") and self.config.rope_parameters:
            mrope_section = self.config.rope_parameters.get("mrope_section")

        if mrope_section is None and self.rope_scaling:
            mrope_section = self.rope_scaling.get("mrope_section", None)

        if mrope_section is None:
            # Fallback: assume 3 equal parts
            mrope_section = [self.head_dim // 3] * 3

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = (
            query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        )
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
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
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    Glm4vTextAttention.forward = patched_forward
