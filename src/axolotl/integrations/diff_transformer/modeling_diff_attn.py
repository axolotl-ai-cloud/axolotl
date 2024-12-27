"""Modeling for differential transformers."""

import logging
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
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

logger = logging.getLogger(__name__)


class LlamaDifferentialConfig(LlamaConfig):
    """Configuration class for Differential LLaMA model."""

    model_type = "llama-differential"

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

    config_class = LlamaDifferentialConfig
    base_model_prefix = "llama_differential"

    def __init__(self, config):
        super().__init__(config)

        # Handle attention implementation
        attn_impl = config._attn_implementation or "eager"
        if attn_impl in config._attn_implementations:
            attn_impl = config._attn_implementations[attn_impl]

        # Validate attention implementation
        valid_impls = [
            None,
            "differential_eager",
            "differential_sdpa",
            "differential_flash_attention_2",
        ]
        if attn_impl not in valid_impls:
            raise ValueError(f"Invalid attention implementation: {attn_impl}")

        # Replace standard attention with differential attention in each layer
        attn_classes = {
            "differential_eager": LlamaDifferentialAttention,
            "differential_sdpa": LlamaDifferentialSdpaAttention,
            "differential_flash_attention_2": LlamaDifferentialFlashAttention2,
        }
        attn_class = attn_classes.get(attn_impl, LlamaDifferentialAttention)

        for idx, layer in enumerate(self.layers):
            layer.self_attn = attn_class(config, idx)

    # pylint: disable=protected-access
    @classmethod
    def _autoset_attn_implementation(
        cls, config, **kwargs
    ):  # pylint: disable=unused-argument
        config._attn_implementation_autoset = True
        attn_implementation = getattr(config, "_attn_implementation", None)

        # Map standard types to differential types if mapping exists
        if attn_implementation in config._attn_implementations:
            config._attn_implementation = config._attn_implementations[
                attn_implementation
            ]
            return config

        # If no mapping, validate it's a valid differential type
        valid_impls = [
            None,
            "differential_eager",
            "differential_sdpa",
            "differential_flash_attention_2",
        ]
        if attn_implementation not in valid_impls:
            message = (
                f"Specified `attn_implementation={attn_implementation}` is not supported. "
                f"The only possible arguments are: {', '.join(repr(x) for x in valid_impls if x)}"
            )
            raise ValueError(message)

        return config

    @classmethod
    def from_llama(
        cls,
        model: Union[LlamaModel, LlamaForCausalLM],
        config: Optional[LlamaDifferentialConfig] = None,
    ) -> "LlamaDifferentialModel":
        """Convert a LlamaModel to use differential attention."""
        logger.info(f"Converting {type(model).__name__} to {cls.__name__}")

        # Handle LlamaForCausalLM
        if isinstance(model, LlamaForCausalLM):
            model = model.model

        if config is None:
            config = LlamaDifferentialConfig(**model.config.__dict__)
            logger.debug(f"Created config: {config}")

        # Validate head counts if using split heads mode
        if config.split_heads:
            if config.num_attention_heads % 2 != 0:
                raise ValueError(
                    f"Number of attention heads ({config.num_attention_heads}) must be even "
                    "when using split_heads=True"
                )
            if config.num_key_value_heads % 2 != 0:
                raise ValueError(
                    f"Number of key/value heads ({config.num_key_value_heads}) must be even "
                    "when using split_heads=True"
                )

        new_model = cls(config)

        # Copy all weights except attention
        logger.debug("Copying embeddings and norm")
        new_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
        new_model.norm.load_state_dict(model.norm.state_dict())

        logger.debug("Copying layer weights")
        for layer_idx, (new_layer, old_layer) in enumerate(
            zip(new_model.layers, model.layers)
        ):
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

            # Get the original projection sizes
            old_q_size = old_layer.self_attn.q_proj.weight.size(0)
            old_k_size = old_layer.self_attn.k_proj.weight.size(0)

            if not config.split_heads:
                logger.debug(
                    f"Layer {layer_idx}: Copying Q/K projections with sizes {old_q_size}, {old_k_size}"
                )
                new_layer.self_attn.q_proj.weight.data[:old_q_size].copy_(
                    old_layer.self_attn.q_proj.weight.data
                )
                new_layer.self_attn.k_proj.weight.data[:old_k_size].copy_(
                    old_layer.self_attn.k_proj.weight.data
                )

                if config.zero_init:
                    logger.debug(f"Layer {layer_idx}: Zero initializing")
                    # Zero out components as needed
                    with torch.no_grad():
                        new_layer.self_attn.q_proj.weight.data[old_q_size:].zero_()
                        new_layer.self_attn.k_proj.weight.data[old_k_size:].zero_()
                        new_layer.self_attn.lambda_q1.zero_()
                        new_layer.self_attn.lambda_k1.zero_()
                        new_layer.self_attn.lambda_q2.zero_()
                        new_layer.self_attn.lambda_k2.zero_()
                        new_layer.self_attn.lambda_init.zero_()

        logger.info("Conversion complete")
        return new_model


class LlamaDifferentialForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with differential attention."""

    config_class = LlamaDifferentialConfig
    base_model_prefix = "llama_differential"

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

        # Validate head counts if using split heads mode
        if config.split_heads:
            if config.num_attention_heads % 2 != 0:
                raise ValueError(
                    f"Number of attention heads ({config.num_attention_heads}) must be even "
                    "when using split_heads=True"
                )
            if config.num_key_value_heads % 2 != 0:
                raise ValueError(
                    f"Number of key/value heads ({config.num_key_value_heads}) must be even "
                    "when using split_heads=True"
                )

        new_model = cls(config)
        new_model.model = LlamaDifferentialModel.from_llama(model.model, config)
        new_model.lm_head.load_state_dict(model.lm_head.state_dict())

        return new_model


def register_diff_attn():
    # Register configs
    AutoConfig.register("llama-differential", LlamaDifferentialConfig)

    # Register models
    AutoModel.register(LlamaDifferentialConfig, LlamaDifferentialModel)
    AutoModelForCausalLM.register(LlamaDifferentialConfig, LlamaDifferentialForCausalLM)
