"""
Gemma3 custom decoder layer using liger fused add rms norm kernels
"""

import sys

import torch
from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
from transformers import Cache, GradientCheckpointingLayer
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3MLP,
    Gemma3RMSNorm,
)


class Gemma3AddRMSNorm(LigerFusedAddRMSNorm):
    """
    Fused add rms norm
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps, casting_mode="gemma")


class Gemma3DecoderLayer(GradientCheckpointingLayer):
    """
    Gemma3 decoder layer using liger fused add rms norm
    """

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3AddRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor | None] | None
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)  # type: ignore

        return outputs  # type: ignore


def patch_gemma3():
    import transformers.models.gemma3.modeling_gemma3

    transformers.models.gemma3.modeling_gemma3.Gemma3DecoderLayer = Gemma3DecoderLayer
    sys.modules["transformers.models.gemma3.modeling_gemma3"].Gemma3DecoderLayer = (
        Gemma3DecoderLayer
    )
