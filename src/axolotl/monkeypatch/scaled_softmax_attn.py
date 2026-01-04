"""
Scaled Softmax (SSMax) attention patch.
SSMax: softmax(scores * log(n))
Ref: https://arxiv.org/abs/2501.19399
"""

import math

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_original_flash_fn = None
_scale_parameters = {}


def patch_scaled_softmax_attention(
    scaling_factor_init: float = 0.168, model: PreTrainedModel = None
):
    """
    Patch Flash Attention to apply Scaled Softmax (SSMax).
    """
    global _original_flash_fn, _scale_parameters

    if model is None:
        raise ValueError("Model must be provided to register learnable parameters")

    for name, module in model.named_modules():
        is_self_attn = hasattr(module, "q_proj") or hasattr(module, "qkv_proj")
        is_attention_named = "self_attn" in name.lower()

        if is_attention_named and is_self_attn:
            scale_param = nn.Parameter(torch.tensor(scaling_factor_init))
            module.register_parameter("ssmax_scale", scale_param)
            _scale_parameters[id(module)] = scale_param
            LOG.info(f"Registered learnable SSMax scale for {name}")

    if "flash_attention_2" in ALL_ATTENTION_FUNCTIONS:
        _original_flash_fn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        def flash_with_ssmax(
            module, query, key, value, attention_mask, scaling=None, **kw
        ):
            if scaling is None:
                scaling = 1.0 / math.sqrt(query.size(-1))

            scale_param = getattr(module, "ssmax_scale", None)
            if scale_param is not None:
                seq_len = query.size(2)
                ssmax_factor = scale_param * math.log(max(seq_len, 2))
                modified_scaling = scaling * ssmax_factor
            else:
                modified_scaling = scaling

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
        LOG.info("Patched flash_attention_2 with learnable SSMax")
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
