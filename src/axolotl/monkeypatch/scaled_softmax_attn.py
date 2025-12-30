"""
Scaled Softmax (SSMax) attention patch.
SSMax: softmax(scores * log(n))
Ref: https://arxiv.org/abs/2501.19399
"""

import math

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_original_flash_fn = None


def patch_scaled_softmax_attention(scaling_factor: float = 1.0, model_type: str = None):
    """
    Patch Flash Attention to apply Scaled Softmax (SSMax).

    Args:
        scaling_factor: Multiplier for the log(n) scaling. Default 1.0.
        model_type: Optional model type string (currently unused, for future extension).
    """
    global _original_flash_fn
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def ssmax_scale(seq_len):
        return scaling_factor * math.log(max(seq_len, 2))

    # Patch flash_attention_2
    if "flash_attention_2" in ALL_ATTENTION_FUNCTIONS:
        _original_flash_fn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        def flash_with_ssmax(
            module, query, key, value, attention_mask, scaling=None, **kw
        ):
            modified_scaling = (scaling or 1.0) * ssmax_scale(query.size(2))
            return _original_flash_fn(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=modified_scaling,
                **kw,
            )

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_with_ssmax
        LOG.info(f"Patched flash_attention_2 with SSMax (factor={scaling_factor})")
    else:
        LOG.warning(
            "SSMax requires flash_attention_2 which is not available. "
            "Please enable flash_attention: true in your config."
        )


def unpatch_scaled_softmax_attention():
    """Restore the original Flash Attention function."""
    global _original_flash_fn
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _original_flash_fn:
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _original_flash_fn
        _original_flash_fn = None
        LOG.info("Unpatched flash_attention_2, restored original")
