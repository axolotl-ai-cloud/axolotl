"""module for building the auto wrap policy for FSDP"""
import functools

from peft import PrefixEncoder, PromptEmbedding, PromptEncoder
from torch import nn
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaMLP,
)
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralDecoderLayer,
    MistralMLP,
)

SUPPORTED_AUTO_WRAP_MODEL_TYPES = [
    "mistral",
    "llama",
]


def get_wrapping_policy_factory(model_type):
    if model_type == "llama":
        attention_classes = LLAMA_ATTENTION_CLASSES
        layer_to_wrap = LlamaDecoderLayer
        model_mlp = LlamaMLP
    elif model_type == "mistral":
        attention_classes = MISTRAL_ATTENTION_CLASSES
        layer_to_wrap = MistralDecoderLayer
        model_mlp = MistralMLP

    def get_wrapping_policy(custom_policy: bool = False):
        """This checks for lora layers (has weight and requires_grad)"""
        if custom_policy:

            def lambda_policy_fn(module):
                # LORA trainable layers.
                return isinstance(module, nn.Sequential) and all(
                    m.weight.requires_grad for m in module
                )

        else:

            def lambda_policy_fn(module):
                return (
                    len(list(module.named_children())) == 0
                    and getattr(module, "weight", None) is not None
                    and module.weight.requires_grad
                )

        def self_attn_policy_fn(module):
            # Check module name is self_attn.
            return isinstance(module, tuple(attention_classes.values()))

        def mlp_policy_fn(module):
            # Check module name is self_attn.
            return isinstance(module, model_mlp)

        lambda_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
        )
        self_attn_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn
        )
        mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
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
        if custom_policy:
            policies.extend([self_attn_policy, mlp_policy])
        return functools.partial(_or_policy, policies=policies)

    return get_wrapping_policy
