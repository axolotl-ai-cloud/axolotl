"""
conversion for llama models to use RALA attention
"""
import logging

from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention

from axolotl.integrations.rala.auto.llama.modeling_rala import (
    LlamaRALAAttention,
    LlamaRalaDecoderLayer,
)

logger = logging.getLogger(__name__)

ATTENTION_MAPPING = {
    LlamaAttention: LlamaRALAAttention,
}


def copy_attention_weights(
    old_attn,
    new_attn,
    zero_init: bool = False,
) -> None:
    """
    Copy weights from old attention layer to new RALA layer.
    Copies q, k, v, o
    """
    new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
    new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
    new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
    new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

    # Zero out lambda parameters for exact equivalence
    if zero_init:
        nn.init.zeros_(new_attn.phi.weight)
    else:
        nn.init.normal_(new_attn.phi.weight)
    if new_attn.phi.bias:
        nn.init.normal_(new_attn.phi.bias)

    logger.debug(
        "Copied positive attention weights from %s to %s",
        type(old_attn).__name__,
        type(new_attn).__name__,
    )


def convert_to_rala(
    model: PreTrainedModel, zero_init: bool = False, softmax_every_n: int = 6
) -> PreTrainedModel:
    """Convert a pre-trained model's attention layers to differential attention"""
    layer_idx = 0

    def convert_module(module, softmax_every, num_hidden_layers):
        nonlocal layer_idx

        # Iterate through module children, convert any attn layers to diff attn
        for name, child in module.named_children():
            if isinstance(child, tuple(ATTENTION_MAPPING.keys())):
                decoder_layer_idx = child.layer_idx
                if LlamaRalaDecoderLayer.is_layer_idx_softmax(
                    num_hidden_layers, decoder_layer_idx, softmax_every
                ):
                    continue
                # Choose appropriate differential attention class
                # pylint: disable=duplicate-code
                attention_class = ATTENTION_MAPPING[type(child)]

                layer_type = type(child).__name__
                logger.info(
                    f"Converting attention layer {decoder_layer_idx}: {layer_type} to {attention_class.__name__}"
                )

                # Create new diff attn layer
                new_attention = attention_class(
                    config=module.config if hasattr(module, "config") else model.config,
                    layer_idx=layer_idx,
                )

                # Copy weights from old attention to new attention
                new_attention.to(child.q_proj.weight.device)
                copy_attention_weights(child, new_attention, zero_init=zero_init)

                # Replace the layer
                setattr(module, name, new_attention)
                layer_idx += 1
            elif len(list(child.children())) > 0:
                convert_module(child, softmax_every, num_hidden_layers)

    model.config.softmax_every = softmax_every_n
    convert_module(model, softmax_every_n, model.config.num_hidden_layers)
    logger.info(f"Converted {layer_idx} attention layers to RALA attention")

    model.config.architectures = [
        "LlamaRalaForCausalLM",
    ]
    model.config.model_type = "llama-rala"
    # model.config.auto_map = {
    #     "AutoConfig": "llama.configuration_rala.LlamaRalaConfig",
    #     "AutoModel": "llama.modeling_rala.LlamaRalaModel",
    #     "AutoModelForCausalLM": "llama.modeling_rala.LlamaRalaForCausalLM",
    # }
    return model
