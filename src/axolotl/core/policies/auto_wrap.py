"""module for building the auto wrap policy for FSDP"""
import functools

from peft import PrefixEncoder, PromptEmbedding, PromptEncoder
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

SUPPORTED_AUTO_WRAP_MODEL_TYPES = [
    "llama",
    "mistral",
    "mixtral",
]


def get_wrapping_policy_factory(model_type):
    if model_type == "llama":
        layer_to_wrap = LlamaDecoderLayer
    elif model_type == "mistral":
        layer_to_wrap = MistralDecoderLayer
    elif model_type == "mixtral":
        layer_to_wrap = MixtralDecoderLayer

    def get_wrapping_policy():
        """This checks for lora layers (has weight and requires_grad)"""

        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
        )
        transformer_layer_name = layer_to_wrap
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(
                PrefixEncoder,
                PromptEncoder,
                PromptEmbedding,
                transformer_layer_name,
            ),
        )
        policies = [lambda_policy, transformer_wrap_policy]
        return functools.partial(_or_policy, policies=policies)

    return get_wrapping_policy
