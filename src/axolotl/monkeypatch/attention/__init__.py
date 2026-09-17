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


def register_xformers_attn():
    """Register xformers as its own attention backend with FA2 mask behavior."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from .xformers import xformers_attention_forward

    ALL_ATTENTION_FUNCTIONS.register("xformers", xformers_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "xformers", ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
    )


def register_sage_attn():
    """Register sage as its own attention backend with FA2 mask behavior."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from .sage_attn import sage_attention_forward

    ALL_ATTENTION_FUNCTIONS.register("sage", sage_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "sage", ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
    )
