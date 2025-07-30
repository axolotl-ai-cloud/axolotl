"""
Custom modeling for Llama for fused rms add kernels
"""

import sys

import torch
from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
from transformers import Cache, GradientCheckpointingLayer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class LlamaAddRMSNorm(LigerFusedAddRMSNorm):
    """
    Fused add rms norm
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps, casting_mode="llama")


class LlamaDecoderLayer(GradientCheckpointingLayer):
    """
    Llama decoder layer using liger fused add rms norm
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaAddRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # pylint: disable=duplicate-code
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def patch_llama():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
    sys.modules["transformers.models.llama.modeling_llama"].LlamaDecoderLayer = (
        LlamaDecoderLayer
    )
