"""
Scaled Softmax (SSMax) attention patch using FlexAttention.
SSMax:  softmax(scores * s * log(n) + b) where n is the position index
Ref: https://arxiv.org/abs/2501.19399
"""

import torch
from transformers import PreTrainedModel

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import (
        compile_friendly_flex_attention,
        repeat_kv,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None

_ssmax_config = {}


def patch_scaled_softmax_attention(
    scaling_factor_init: float = 0.43, bias: float = 0.0, model: PreTrainedModel = None
):
    """Patch attention to apply SSMax via FlexAttention score_mod."""
    global _ssmax_config

    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("SSMax requires FlexAttention.")

    _ssmax_config["ssmax_s"] = scaling_factor_init
    _ssmax_config["ssmax_b"] = bias

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if "flex_attention" in ALL_ATTENTION_FUNCTIONS:
        _ssmax_config["original_flex_fn"] = ALL_ATTENTION_FUNCTIONS["flex_attention"]
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = ssmax_flex_attention_forward
        LOG.info(
            f"Patched flex_attention with SSMax (s={scaling_factor_init}, b={bias})"
        )
    else:
        LOG.warning("flex_attention not found.  Ensure flex_attention:  true is set.")


def ssmax_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FlexAttention forward with SSMax:  score * (s * log(n) + b)."""

    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError("flex_attention does not support dropout")

    ssmax_s = _ssmax_config.get("ssmax_s", 0.43)
    ssmax_b = _ssmax_config.get("ssmax_b", 0.0)

    position_ids = kwargs.get("position_ids", None)
    position_ids_flat = position_ids.view(-1) if position_ids is not None else None

    block_mask = attention_mask if isinstance(attention_mask, BlockMask) else None
    score_mask = None if block_mask else attention_mask

    if score_mask is not None:
        score_mask = score_mask[:, :, :, : key.shape[-2]]

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        """
        Apply SSMax scaling:  score * (s * log(n) + b)
        where n is the relative position within each packed sequence.
        """
        if position_ids_flat is not None:
            relative_pos = position_ids_flat[q_idx]
            n = (relative_pos + 1).float()
        else:
            n = (q_idx + 1).float()

        n = torch.clamp(n, min=2.0)

        ssmax_scale = ssmax_s * torch.log(n) + ssmax_b
        score = score * ssmax_scale

        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)

        if score_mask is not None:
            score = score + score_mask[batch_idx][0][q_idx][kv_idx]

        return score

    enable_gqa = True
    if (query.shape[1] & (query.shape[1] - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    return_lse = query.device.type != "cpu"
    flex_output = compile_friendly_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kwargs.get("kernel_options"),
        return_lse=return_lse,
        training=module.training,
    )

    if return_lse:
        attention_output, lse = flex_output
        lse = lse.to(value.dtype)
    else:
        attention_output, lse = flex_output, None

    return attention_output.transpose(1, 2).contiguous(), lse


def unpatch_scaled_softmax_attention():
    """Restore the original FlexAttention function."""
    global _ssmax_config
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if "original_flex_fn" in _ssmax_config:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = _ssmax_config["original_flex_fn"]
        _ssmax_config.clear()
        LOG.info("Unpatched flex_attention, restored original")
