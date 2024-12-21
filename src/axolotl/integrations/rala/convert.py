"""
conversion for llama models to use RALA attention
"""
import logging

from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention

from axolotl.integrations.rala import LlamaRALAAttention

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
        nn.init.normal_(new_attn.phi)
    nn.init.zeros_(new_attn.phi.bias)

    logger.debug(
        "Copied positive attention weights from %s to %s",
        type(old_attn).__name__,
        type(new_attn).__name__,
    )


def convert_to_rala(
    model: PreTrainedModel,
    zero_init: bool = False,
) -> PreTrainedModel:
    """Convert a pre-trained model's attention layers to differential attention"""
    layer_idx = 0

    def convert_module(module):
        nonlocal layer_idx

        # Iterate through module children, convert any attn layers to diff attn
        for name, child in module.named_children():
            if isinstance(child, tuple(ATTENTION_MAPPING.keys())):
                # Choose appropriate differential attention class
                # pylint: disable=duplicate-code
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
                copy_attention_weights(child, new_attention, zero_init=zero_init)

                # Replace the layer
                setattr(module, name, new_attention)
                layer_idx += 1
            elif len(list(child.children())) > 0:
                convert_module(child)

    convert_module(model)
    logger.info(f"Converted {layer_idx} attention layers to RALA attention")

    return model
