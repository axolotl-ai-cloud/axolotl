# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Linear LLaMA model implementation."""


from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from axolotl.utils.dict import DictDefault

from .attention import LolcatsLinearAttention
from .configuration_linear_llama import LinearLlamaConfig


class LinearLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Modified LlamaDecoderLayer that uses LinearAttention instead of standard attention.
    """

    def __init__(self, config: LinearLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the attention layer with our custom attention
        self.self_attn = LolcatsLinearAttention(
            base_attn=self.self_attn,  # type: ignore
            layer_idx=layer_idx,
            **config.attention_config,
        )


class LinearLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LinearLlamaDecoderLayer`]

    Args:
        config: LinearLlamaConfig
    """

    config_class = LinearLlamaConfig
    base_model_prefix = "linear_llama"

    def __init__(self, config: LinearLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LinearLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class LinearLlamaForCausalLM(LlamaForCausalLM):
    """
    Linear LLaMA model for causal language modeling.
    """

    config_class = LinearLlamaConfig
    base_model_prefix = "linear_llama"

    def __init__(self, config):
        super().__init__(config)
        self.model = LinearLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_llama(
        cls,
        model: LlamaModel | LlamaForCausalLM,
        config: LinearLlamaConfig,
        train_attention: bool = False,
        remove_base_attn: bool = True,
    ) -> "LinearLlamaForCausalLM":
        """
        Initialize a LinearLlamaForCausalLM from a LlamaModel
        """

        # Handle LlamaForCausalLM
        if isinstance(model, LlamaForCausalLM):
            llama_model = model.model
        else:
            llama_model = model

        if config is None:
            raise ValueError("Missing config")

        from axolotl.integrations.lolcats.linearize_attention import convert_attention

        llama_model = convert_attention(
            llama_model,
            DictDefault(**config.attention_config),
            train_attention=train_attention,
            remove_base_attn=remove_base_attn,
        )

        # initialize the model with prior weights
        new_model = cls(config=config)
        del new_model.model  # remove the default model
        del new_model.lm_head  # remove the default lm_head
        new_model.model = llama_model
        new_model.lm_head = model.lm_head

        return new_model

    def toggle_attention(self, train: bool = True):
        """
        Toggle attention to be trainable or not
        """
        from axolotl.integrations.lolcats.linearize_attention import toggle_attention

        toggle_attention(self.model, train=train)

    def remove_base_attention(self):
        """
        Remove base attention after distillation
        """
        from axolotl.integrations.lolcats.linearize_attention import (
            remove_base_attention,
        )

        remove_base_attention(self.model)
