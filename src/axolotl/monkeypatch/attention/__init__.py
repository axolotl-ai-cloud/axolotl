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


def register_sparse_attn():
    """Register nsa/fsa names so transformers accepts them at load time.

    The actual computation is a model-specific module swap
    (:func:`axolotl.monkeypatch.attention.sparse_attn.patch_sparse_attention`);
    the registered forward is a stub that must never be invoked.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from .sparse_attn import sparse_attention_stub

    for name in ("nsa", "fsa"):
        ALL_ATTENTION_FUNCTIONS.register(name, sparse_attention_stub)
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            name, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
        )
