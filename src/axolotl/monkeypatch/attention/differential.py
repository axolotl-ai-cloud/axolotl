"""Patches related to differential transformers implementation."""

from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES

from axolotl.integrations.diff_transformer.diff_attn import (
    LlamaDifferentialAttention,
    LlamaDifferentialFlashAttention2,
    LlamaDifferentialSdpaAttention,
)


def patch_llama_attention_classes():
    """Patch transformers to support differential attention"""
    # Add our attention class to the registry
    LLAMA_ATTENTION_CLASSES["differential_eager"] = LlamaDifferentialAttention
    LLAMA_ATTENTION_CLASSES["differential_sdpa"] = LlamaDifferentialSdpaAttention
    LLAMA_ATTENTION_CLASSES[
        "differential_flash_attention_2"
    ] = LlamaDifferentialFlashAttention2

    @classmethod
    def new_autoset(_, config, **kwargs):  # pylint: disable=unused-argument
        config._attn_implementation_autoset = True  # pylint: disable=protected-access
        attn_implementation = getattr(config, "_attn_implementation", None)

        valid_impls = [
            None,
            "eager",
            "sdpa",
            "flash_attention_2",
            "differential_eager",
            "differential_sdpa",
            "differential_flash_attention_2",
        ]
        if attn_implementation not in valid_impls:
            message = (
                f"Specified `attn_implementation={attn_implementation}` is not supported. "
                f"The only possible arguments are: {', '.join(repr(x) for x in valid_impls if x)}"
            )
            raise ValueError(message + ".")

        return config

    # Apply patch
    PreTrainedModel._autoset_attn_implementation = (  # pylint: disable=protected-access
        new_autoset
    )
