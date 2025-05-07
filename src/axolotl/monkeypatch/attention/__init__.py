"""
attention module for attention monkeypatches
"""

from transformers.integrations.flash_attention import flash_attention_forward


def patch_xformers_attn_over_fa2():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from .xformers import xformers_attention_forward

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = xformers_attention_forward


def unpatch_xformers_attn_over_fa2():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward()
