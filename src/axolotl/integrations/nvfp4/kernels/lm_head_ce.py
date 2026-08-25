"""Head-aware fused lm_head + cross-entropy dispatcher.

Routes by lm_head dtype to nvfp4_/bf16_fused_ce. ``auto`` picks fp4 else bf16.
"""

from __future__ import annotations

from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _output_head(model: nn.Module):
    """Resolve the ForCausalLM output embedding, unwrapping a PEFT wrapper."""
    causal = model
    if hasattr(model, "get_base_model"):
        try:
            causal = model.get_base_model()
        except Exception:  # pylint: disable=broad-except
            causal = model
    try:
        return causal.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        return None


def _is_fp4_head(lm_head) -> bool:
    from axolotl.integrations.nvfp4.kernels.nvfp4_fused_ce import _nvfp4_lm_head_store

    return _nvfp4_lm_head_store(lm_head) is not None


def _is_plain_frozen_linear(lm_head) -> bool:
    return (
        type(lm_head) is nn.Linear
        and lm_head.bias is None
        and not lm_head.weight.requires_grad
    )


def patch_lm_head_cross_entropy(
    model: nn.Module,
    mode: str,
    vocab_block: int | None = None,
) -> str | None:
    """Patch the ForCausalLM forward with the head-appropriate fused CE kernel.

    Returns the installed kernel (``"fp4"``/``"bf16"``) or ``None`` when nothing
    matched (caller keeps the materialized path).
    """
    if not mode or mode == "off":
        return None

    head = _output_head(model)
    if head is None:
        LOG.warning(
            "lm_head_cross_entropy: model has no output embeddings; keeping the "
            "materialized CE path."
        )
        return None

    resolved = mode
    if mode == "auto":
        if _is_fp4_head(head):
            resolved = "fp4"
        elif _is_plain_frozen_linear(head):
            resolved = "bf16"
        else:
            LOG.warning(
                "lm_head_cross_entropy: auto found neither an NVFP4-packed nor a "
                "plain frozen bias-free nn.Linear lm_head (%s); keeping the "
                "materialized CE path.",
                type(head).__name__,
            )
            return None

    if resolved == "fp4":
        from axolotl.integrations.nvfp4.kernels.nvfp4_fused_ce import (
            patch_model_fused_fp4_ce,
        )

        ok = patch_model_fused_fp4_ce(model, vocab_block=vocab_block)
        return "fp4" if ok else None

    if resolved == "bf16":
        from axolotl.integrations.nvfp4.kernels.bf16_fused_ce import (
            patch_model_bf16_lm_head_cross_entropy,
        )

        ok = patch_model_bf16_lm_head_cross_entropy(model, vocab_block=vocab_block)
        return "bf16" if ok else None

    return None
