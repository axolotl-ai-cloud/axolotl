"""Head-aware fused lm_head + cross-entropy dispatcher.

Routes to the matching tiled-CE kernel by lm_head dtype: ``fp4`` (NVFP4-packed
store) -> nvfp4_fused_ce, ``bf16``/``fp8`` (plain frozen bias-free nn.Linear) ->
bf16_fused_ce / fp8_fused_ce. ``auto`` picks fp4 for an FP4 head else bf16; fp8 is
never auto-selected (parity caveats), so it is opt-in only.
"""

from __future__ import annotations

from typing import Literal

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


def _is_fp4_head(lm_head, fp4_matmul: bool | None) -> bool:
    from axolotl.kernels.nvfp4_fused_ce import (
        _fp4_scaled_mm_enabled,
        _nvfp4_lm_head_fp4_store,
        _nvfp4_lm_head_store,
    )

    if _nvfp4_lm_head_store(lm_head) is not None:
        return True
    return (
        _fp4_scaled_mm_enabled(fp4_matmul)
        and _nvfp4_lm_head_fp4_store(lm_head) is not None
    )


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
    fp4_matmul: bool | None = None,
    granularity: Literal["tensorwise", "rowwise"] = "rowwise",
) -> str | None:
    """Patch the ForCausalLM forward with the head-appropriate fused CE kernel.

    Returns the installed kernel (``"fp4"``/``"bf16"``/``"fp8"``) or ``None`` when
    nothing matched (caller keeps the materialized path).
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
        if _is_fp4_head(head, fp4_matmul):
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
        from axolotl.kernels.nvfp4_fused_ce import patch_model_fused_fp4_ce

        ok = patch_model_fused_fp4_ce(
            model, fp4_matmul=fp4_matmul, vocab_block=vocab_block
        )
        return "fp4" if ok else None

    if resolved == "bf16":
        from axolotl.kernels.bf16_fused_ce import (
            patch_model_bf16_lm_head_cross_entropy,
        )

        ok = patch_model_bf16_lm_head_cross_entropy(model, vocab_block=vocab_block)
        return "bf16" if ok else None

    if resolved == "fp8":
        from axolotl.kernels.fp8_fused_ce import (
            patch_model_fp8_lm_head_cross_entropy,
        )

        ok = patch_model_fp8_lm_head_cross_entropy(
            model, granularity=granularity, vocab_block=vocab_block
        )
        return "fp8" if ok else None

    return None
