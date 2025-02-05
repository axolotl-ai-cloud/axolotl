# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Linear LLaMA model implementation."""

import logging
from functools import partial
from typing import Any

from torch import nn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from .attention import LolcatsLinearAttention
from .configuration_linear_llama import LinearLlamaConfig

LOG = logging.getLogger(__name__)


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
        model: LlamaForCausalLM,
        config: LinearLlamaConfig,
        train_attention: bool = False,
        remove_base_attn: bool = True,
    ) -> "LinearLlamaForCausalLM":
        """
        Initialize a LinearLlamaForCausalLM from a LlamaModel
        """

        if config is None:
            raise ValueError("Missing config")

        # initialize the model with prior weights
        new_model = cls(config=config)

        # remove the default model and lm_head
        del new_model.model
        del new_model.lm_head

        new_model.model = convert_attention(
            model.model,
            attention_config=config.attention_config,
            train_attention=train_attention,
            remove_base_attn=remove_base_attn,
        )
        new_model.lm_head = model.lm_head

        return new_model

    def toggle_attention(self, train: bool = True):
        """
        Toggle attention to be trainable or not
        """

        toggle_attention(self.model, train=train)

    def remove_base_attention(self):
        """
        Remove base attention after distillation
        """

        remove_base_attention(self.model)


def convert_attention(
    model: nn.Module,
    attention_config: dict,
    train_attention: bool = False,
    remove_base_attn: bool = True,
):
    """
    Call to convert all attention layers
    """
    # Get the layers to convert if provided
    softmax_attns = attention_config.get("softmax_attentions", [])

    # Get the attention to convert to
    attention_type = attention_config.get("attention_type")

    if attention_type != "softmax":
        layers = traverse_layers(model)
        for layer_idx, layer in enumerate(
            tqdm(layers, desc="Converting attentions...")
        ):
            if layer_idx not in softmax_attns:
                layer.self_attn = convert_llama_attention(
                    layer,
                    attention_config,
                    layers,
                    train_attention,
                    remove_base_attn,
                )
                layer.self_attn.converted = True
            else:
                # Freeze any preserved softmax attention layers
                for p in layer.parameters():
                    p.requires_grad = False
    else:
        LOG.info(
            f"-> attention_config.attention_type is {attention_type}; not converting attentions"
        )
    return model


def toggle_attention(llama_model: nn.Module, train: bool = False):
    """
    Make attentions trainable if train is True
    -> Set train_attention = False when finetuning
    """
    for layer in traverse_layers(llama_model):
        layer.self_attn.train_attention = train
    return llama_model


def remove_base_attention(llama_model: nn.Module):
    """
    Remove teacher attention after distillation (if we keep it)
    """
    for layer in traverse_layers(llama_model):
        if getattr(layer.self_attn, "base_attn", False):
            del layer.self_attn.base_attn
    return llama_model


def traverse_layers(model: nn.Module, verbose: bool = False):
    """
    Return list of model layers
    """
    try:
        layers = model.model.layers
        if verbose:
            LOG.info("-> Loading from model.model.layers")
    except AttributeError as e:  # if base model
        if verbose:
            LOG.info(e)
        try:
            layers = model.layers
            if verbose:
                LOG.info("-> Loading from model.layers")
        except AttributeError as e1:  # If we make a PEFT model
            if verbose:
                LOG.info(e1)
            layers = model.base_model.model.model.layers
            if verbose:
                LOG.info("-> Loading from model.base_model.model.model.layers")
    return layers


def convert_llama_attention(
    layer: nn.Module,
    attention_config: dict,
    layers: list[nn.Module],  # list of layers
    train_attention: bool = False,
    remove_base_attn: bool = True,
):
    """
    Converts a single layer's attention layer as specified by attention_config
    """
    return get_attention(**attention_config)(
        base_attn=layer.self_attn,
        layer_idx=layer.self_attn.layer_idx,  # Transformers v4.36
        max_layer_idx=len(layers) - 1,
        train_attention=train_attention,
        remove_base_attn=remove_base_attn,
    )


def get_attention(attention_type: str, **kwargs):
    """
    Get the linear attention class; either purely linear or linear with sliding window
    -> 'linear' == 'lolcats_llama'
    -> 'linear and sliding_window' == 'lolcats_llama_window_*'
    """
    kwargs["attention_type"] = attention_type

    if attention_type == "lolcats_llama":
        from .attention import LolcatsLinearAttention

        return partial(LolcatsLinearAttention, **kwargs)

    elif attention_type == "lolcats_llama_window_tk":
        from .attention import LolcatsTKWindowAttention

        return partial(LolcatsTKWindowAttention, **kwargs)

    elif attention_type == "lolcats_llama_window_sw":
        from .attention import LolcatsSlidingWindowAttention

        return partial(LolcatsSlidingWindowAttention, **kwargs)

    elif attention_type == "lolcats_llama_window_sw_linear":
        from .attention import LolcatsLinearSlidingWindowAttention

        return partial(LolcatsLinearSlidingWindowAttention, **kwargs)

    # Experimental chunked linear attentions below
    elif attention_type == "lolcats_long_llama_window_tk":
        from .attention import LolcatsTKWindowLongAttention

        return partial(LolcatsTKWindowLongAttention, **kwargs)

    elif attention_type == "lolcats_long_llama_window_sw":
        from .attention import LolcatsSlidingWindowLongAttention

        return partial(LolcatsSlidingWindowLongAttention, **kwargs)

    # TK generation build (requires Thunderkittens)
    elif attention_type == "lolcats_llama_window_tk_gen":
        from .attention import LolcatsWindowAttentionTKGen

        return partial(LolcatsWindowAttentionTKGen, **kwargs)

    else:
        LOG.info(f"-> attention_type {attention_type} not handled... returning None")
        return None


def get_attention_cache(attention_type: str, past_key_values: Any = None):
    """
    Determine how we store past keys and values when generating
    """
    if attention_type is None:
        return past_key_values

    # LOG.info(f'Returning attention cache based on attention_type == {attention_type}')
    elif "lolcats_llama_window_tk_gen" in attention_type:
        from .attention import LinearAttentionTKWindowGenerationCache

        return LinearAttentionTKWindowGenerationCache()

    elif "llama_window_tk" in attention_type:
        from .attention import LinearAttentionTKWindowCache

        return LinearAttentionTKWindowCache()

    elif "llama_window_sw" in attention_type:
        from .attention import LinearAttentionSlidingWindowCache

        return LinearAttentionSlidingWindowCache()

    elif "llama_window_sw_linear" in attention_type:
        from .attention import LinearAttentionSlidingWindowCache

        return LinearAttentionSlidingWindowCache()

    # TK generation build (requires Thunderkittens)
    elif attention_type == "lolcats_llama_window_tk_gen":
        from .attention import LinearAttentionTKWindowGenerationCache

        return LinearAttentionTKWindowGenerationCache()

    elif "softmax" in attention_type:
        return past_key_values

    else:
        from .attention import LinearAttentionState

        return LinearAttentionState()


def register_linear_llama():
    """
    Register Linear LLaMA model with the Transformers library.
    """

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("linear_llama", LinearLlamaConfig)
    AutoModel.register(LinearLlamaConfig, LinearLlamaModel)
    AutoModelForCausalLM.register(LinearLlamaConfig, LinearLlamaForCausalLM)
