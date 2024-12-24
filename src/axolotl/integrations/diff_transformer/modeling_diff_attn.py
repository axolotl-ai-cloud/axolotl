"""Modeling for differential transformers."""

from typing import Optional

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)

from .diff_attn import (
    LlamaDifferentialAttention,
    LlamaDifferentialAttentionBase,
    LlamaDifferentialFlashAttention2,
    LlamaDifferentialSdpaAttention,
)


class LlamaDifferentialConfig(LlamaConfig):
    """Configuration class for Differential LLaMA model."""

    def __init__(
        self,
        split_heads: bool = False,
        sublayer_norm: bool = True,
        zero_init: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_heads = split_heads
        self.sublayer_norm = sublayer_norm
        self.zero_init = zero_init
        self.architectures = ["LlamaDifferentialModel"]
        self._attn_implementations = {
            "eager": "differential_eager",
            "sdpa": "differential_sdpa",
            "flash_attention_2": "differential_flash_attention_2",
        }


class LlamaDifferentialPreTrainedModel(LlamaPreTrainedModel):
    """Base class for differential LLaMA models."""

    config_class = LlamaDifferentialConfig
    base_model_prefix = "llama_differential"

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (LlamaDifferentialAttentionBase, LlamaModel)):
            module.gradient_checkpointing = value


class LlamaDifferentialModel(LlamaModel):
    """LlamaModel with differential attention."""

    def __init__(self, config):
        super().__init__(config)
        # Replace standard attention with differential attention in each layer
        for layer in self.layers:
            attn_impl = config._attn_implementation or "eager"
            if attn_impl == "eager":
                layer.self_attn = LlamaDifferentialAttention(config, layer.layer_idx)
            elif attn_impl == "sdpa":
                layer.self_attn = LlamaDifferentialSdpaAttention(
                    config, layer.layer_idx
                )
            elif attn_impl == "flash_attention_2":
                layer.self_attn = LlamaDifferentialFlashAttention2(
                    config, layer.layer_idx
                )

    @classmethod
    def from_llama(
        cls, model: LlamaModel, config: Optional[LlamaDifferentialConfig] = None
    ) -> "LlamaDifferentialModel":
        """Convert a LlamaModel to use differential attention."""
        if config is None:
            config = LlamaDifferentialConfig(**model.config.__dict__)

        new_model = cls(config)
        # Copy all weights except attention
        new_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
        new_model.norm.load_state_dict(model.norm.state_dict())

        for new_layer, old_layer in zip(new_model.layers, model.layers):
            # Copy everything except attention weights
            new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())
            new_layer.input_layernorm.load_state_dict(
                old_layer.input_layernorm.state_dict()
            )
            new_layer.post_attention_layernorm.load_state_dict(
                old_layer.post_attention_layernorm.state_dict()
            )

            # Handle attention weights
            new_layer.self_attn.v_proj.load_state_dict(
                old_layer.self_attn.v_proj.state_dict()
            )
            new_layer.self_attn.o_proj.load_state_dict(
                old_layer.self_attn.o_proj.state_dict()
            )

            if config.split_heads:
                new_layer.self_attn.q_proj.weight.data.copy_(
                    old_layer.self_attn.q_proj.weight.data
                )
                new_layer.self_attn.k_proj.weight.data.copy_(
                    old_layer.self_attn.k_proj.weight.data
                )
            else:
                new_layer.self_attn.q_proj.weight.data[: config.hidden_size].copy_(
                    old_layer.self_attn.q_proj.weight.data
                )
                new_layer.self_attn.k_proj.weight.data[: config.hidden_size].copy_(
                    old_layer.self_attn.k_proj.weight.data
                )

                if config.zero_init:
                    # Zero out components as needed
                    with torch.no_grad():
                        new_layer.self_attn.q_proj.weight.data[
                            config.hidden_size :
                        ].zero_()
                        new_layer.self_attn.k_proj.weight.data[
                            config.hidden_size :
                        ].zero_()
                        new_layer.self_attn.lambda_q1.zero_()
                        new_layer.self_attn.lambda_k1.zero_()
                        new_layer.self_attn.lambda_q2.zero_()
                        new_layer.self_attn.lambda_k2.zero_()
                        new_layer.self_attn.lambda_init.zero_()

        return new_model


class LlamaDifferentialForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with differential attention."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaDifferentialModel(config)

    @classmethod
    def from_llama(
        cls, model: LlamaForCausalLM, config: Optional[LlamaDifferentialConfig] = None
    ) -> "LlamaDifferentialForCausalLM":
        """Convert a LlamaForCausalLM to use differential attention."""
        if config is None:
            config = LlamaDifferentialConfig(**model.config.__dict__)

        new_model = cls(config)
        new_model.model = LlamaDifferentialModel.from_llama(model.model, config)
        new_model.lm_head.load_state_dict(model.lm_head.state_dict())
        return new_model
