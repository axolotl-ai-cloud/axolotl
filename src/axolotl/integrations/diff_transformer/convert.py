"""Differential attention conversion logic for a huggingface pre-trained model."""
import logging
from typing import Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
)

from .diff_attn import (
    LlamaDifferentialAttention,
    LlamaDifferentialFlashAttention2,
    LlamaDifferentialSdpaAttention,
)

logger = logging.getLogger(__name__)

ATTENTION_MAPPING = {
    LlamaAttention: LlamaDifferentialAttention,
    LlamaSdpaAttention: LlamaDifferentialSdpaAttention,
    LlamaFlashAttention2: LlamaDifferentialFlashAttention2,
}


def copy_attention_weights(
    old_attn: Union[LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2],
    new_attn: Union[
        LlamaDifferentialAttention,
        LlamaDifferentialSdpaAttention,
        LlamaDifferentialFlashAttention2,
    ],
    zero_init: bool = False,
) -> None:
    """
    Copy weights from old attention layer to new differential attention layer.
    Copies old weights to Q1 and K1, zeros out Q2 and K2 for exact equivalence
    to original attention mechanism.
    """
    # For Q projection (Q1 and Q2)
    new_q = torch.empty_like(new_attn.q_proj.weight.data)
    new_q[: new_attn.hidden_size] = old_attn.q_proj.weight.data  # Q1
    if zero_init:
        new_q[new_attn.hidden_size :] = 0
    else:
        nn.init.normal_(new_q[new_attn.hidden_size :], mean=0, std=0.1)
    new_attn.q_proj.weight.data.copy_(new_q)

    # For K projection (K1 and K2)
    old_kv_size = old_attn.k_proj.weight.data.size(0)  # Size for 3 heads
    new_k = torch.empty_like(new_attn.k_proj.weight.data)
    new_k[:old_kv_size] = old_attn.k_proj.weight.data  # K1
    if zero_init:
        new_k[old_kv_size:] = 0
    else:
        nn.init.normal_(new_k[old_kv_size:], mean=0, std=0.1)
    new_attn.k_proj.weight.data.copy_(new_k)

    # For V projection (single V)
    new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)

    # Output projection remains the same
    new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

    # Zero out lambda parameters for exact equivalence
    if zero_init:
        nn.init.zeros_(new_attn.lambda_q1)
        nn.init.zeros_(new_attn.lambda_k1)
        nn.init.zeros_(new_attn.lambda_q2)
        nn.init.zeros_(new_attn.lambda_k2)
        nn.init.zeros_(new_attn.lambda_init)

    logger.debug(
        "Copied positive attention weights from %s to %s",
        type(old_attn).__name__,
        type(new_attn).__name__,
    )


def convert_to_diff_attn(
    model: PreTrainedModel,
    zero_init: bool = False,
    sublayer_norm: bool = True,
    split_heads: bool = True,
) -> PreTrainedModel:
    """Convert a pre-trained model's attention layers to differential attention"""
    layer_idx = 0

    # Set sublayer norm as config on the model.
    model.config.sublayer_norm = sublayer_norm
    model.config.split_heads = split_heads

    def convert_module(module):
        nonlocal layer_idx

        # Iterate through module children, convert any attn layers to diff attn
        for name, child in module.named_children():
            if isinstance(child, tuple(ATTENTION_MAPPING.keys())):
                # Choose appropriate differential attention class
                attention_class = ATTENTION_MAPPING[type(child)]

                layer_type = type(child).__name__
                logger.info(
                    f"Converting attention layer {layer_idx}: {layer_type} to {attention_class.__name__}"
                )

                # Create new diff attn layer
                new_attention = attention_class(
                    config=module.config if hasattr(module, "config") else model.config,
                    layer_idx=layer_idx,
                )

                # Copy weights from old attention to new attention
                new_attention.to(child.q_proj.weight.device)
                if not split_heads:
                    copy_attention_weights(child, new_attention, zero_init=zero_init)

                # Replace the layer
                setattr(module, name, new_attention)
                layer_idx += 1
            elif len(list(child.children())) > 0:
                convert_module(child)

    convert_module(model)
    logger.info(f"Converted {layer_idx} attention layers to differential attention")

    return model
