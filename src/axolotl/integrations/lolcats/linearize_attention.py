"""
Convert attention to linear attention

Adapted from: https://github.com/HazyResearch/lolcats/blob/main/src/model/convert_model.py

@misc{zhang2024lolcatslowranklinearizinglarge,
      title={LoLCATs: On Low-Rank Linearizing of Large Language Models},
      author={Michael Zhang and Simran Arora and Rahul Chalamala and Alan Wu and Benjamin Spector and Aaryan Singhal and Krithik Ramesh and Christopher Ré},
      year={2024},
      eLOG.info={2410.10254},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10254},
}
"""

import logging
from functools import partial
from typing import Any

import torch.nn as nn
from tqdm import tqdm

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.integrations.lolcats.linearize_attention")


def convert_attention(
    model: nn.Module,
    attention_config: DictDefault,
    train_attention: bool = False,
    remove_base_attn: bool = True,
):
    """
    Call to convert all attention layers
    """
    softmax_attns = []
    if "softmax_attentions" in attention_config:
        softmax_attns = attention_config["softmax_attentions"]
    if attention_config.attention_type != "softmax":
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
            else:  # Freeze any preserved softmax attention layers
                for p in layer.parameters():
                    p.requires_grad = False
    else:
        LOG.info(
            f"-> attention_config.attention_type is {attention_config.attention_type}; not converting attentions"
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
    attention_config: DictDefault,
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
        from .linear_llama.attention import LolcatsLinearAttention

        return partial(LolcatsLinearAttention, **kwargs)

    elif attention_type == "lolcats_llama_window_tk":
        from .linear_llama.attention import LolcatsTKWindowAttention

        return partial(LolcatsTKWindowAttention, **kwargs)

    elif attention_type == "lolcats_llama_window_sw":
        from .linear_llama.attention import LolcatsSlidingWindowAttention

        return partial(LolcatsSlidingWindowAttention, **kwargs)

    elif attention_type == "lolcats_llama_window_sw_linear":
        from .linear_llama.attention import LolcatsLinearSlidingWindowAttention

        return partial(LolcatsLinearSlidingWindowAttention, **kwargs)

    # Experimental chunked linear attentions below
    elif attention_type == "lolcats_long_llama_window_tk":
        from .linear_llama.attention import LolcatsTKWindowLongAttention

        return partial(LolcatsTKWindowLongAttention, **kwargs)

    elif attention_type == "lolcats_long_llama_window_sw":
        from .linear_llama.attention import LolcatsSlidingWindowLongAttention

        return partial(LolcatsSlidingWindowLongAttention, **kwargs)

    # TK generation build (requires Thunderkittens)
    elif attention_type == "lolcats_llama_window_tk_gen":
        from .linear_llama.attention import LolcatsWindowAttentionTKGen

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
        from .linear_llama.attention import LinearAttentionTKWindowGenerationCache

        return LinearAttentionTKWindowGenerationCache()

    elif "llama_window_tk" in attention_type:
        from .linear_llama.attention import LinearAttentionTKWindowCache

        return LinearAttentionTKWindowCache()

    elif "llama_window_sw" in attention_type:
        from .linear_llama.attention import LinearAttentionSlidingWindowCache

        return LinearAttentionSlidingWindowCache()

    elif "llama_window_sw_linear" in attention_type:
        from .linear_llama.attention import LinearAttentionSlidingWindowCache

        return LinearAttentionSlidingWindowCache()

    # TK generation build (requires Thunderkittens)
    elif attention_type == "lolcats_llama_window_tk_gen":
        from .linear_llama.attention import LinearAttentionTKWindowGenerationCache

        return LinearAttentionTKWindowGenerationCache()

    elif "softmax" in attention_type:
        return past_key_values

    else:
        from .linear_llama.attention import LinearAttentionState

        return LinearAttentionState()
