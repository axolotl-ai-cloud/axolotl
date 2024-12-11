"""Differential attention conversion logic for a huggingface pre-trained model."""
import logging

from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.mixtral.modeling_mixtral import MixtralAttention

from .multihead_diffattn import DifferentialAttention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_diff_attention(model: PreTrainedModel) -> PreTrainedModel:
    """Convert a pre-trained model's attention layers to differential attention"""
    attention_patterns = (LlamaAttention, MistralAttention, MixtralAttention)
    layer_idx = 0

    # Get model dtype from existing weights
    model_dtype = next(model.parameters()).dtype

    def convert_module(module):
        nonlocal layer_idx

        # Iterate through module children, convert any attn layers to diff attn
        for name, child in module.named_children():
            if isinstance(child, attention_patterns):
                layer_type = type(child).__name__
                logger.info(f"Converting attention layer {layer_idx}: {layer_type}")

                # Create new diff attn layer
                new_attention = DifferentialAttention(
                    config=module.config if hasattr(module, "config") else model.config,
                    layer_idx=layer_idx,
                    dtype=model_dtype,
                )

                # Replace the layer
                setattr(module, name, new_attention)
                layer_idx += 1
            elif len(list(child.children())) > 0:
                convert_module(child)

    convert_module(model)
    logger.info(f"Converted {layer_idx} attention layers to differential attention")

    return model
