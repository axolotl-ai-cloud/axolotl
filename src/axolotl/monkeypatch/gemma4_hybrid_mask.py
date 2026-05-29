"""Hybrid attention mask fix for Gemma 4.

Gemma 4 has full-attention (global) layers with ``head_dim=512`` which
exceeds flash-attention-2's supported size. Axolotl's hybrid-attention
patch in ``patch_manager._apply_gemma_hybrid_attention`` works around
this by forcing ``_attn_implementation="sdpa"`` on each global layer's
``self_attn.config``, leaving sliding-window layers on FA2.

The per-layer config override alone is insufficient, however:
``Gemma4TextModel.forward`` builds a single ``causal_mask_mapping`` dict
using the **model-level** config and passes the mapped mask to each
decoder layer. With FA2 still set at the model level, the ``full_attention``
entry in that mapping is a 2D mask (FA2 format), but SDPA needs a 4D mask.
The global layers then fail with::

    RuntimeError: The expanded size of the tensor (S) must match the existing
    size (B) at non-singleton dimension 2. Target sizes: [B, H, S, S]. Tensor
    sizes: [B, S]

...when the sequence length grows past roughly 7k tokens.

This module fixes the symptom by monkey-patching ``create_causal_mask`` in
``transformers.models.gemma4.modeling_gemma4``'s module namespace — NOT
the original in ``masking_utils``. The wrapper forces
``_attn_implementation="sdpa"`` on a shallow-copied config before calling
through, so the ``full_attention`` mask built inside ``Gemma4TextModel.forward``
is always 4D/SDPA-compatible. ``create_sliding_window_causal_mask`` is left
alone, so sliding-window layers continue to receive FA2-format masks.

The patch is idempotent. Install once per process, before any Gemma 4
forward pass runs.
"""

from __future__ import annotations

import copy
from typing import Any

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_APPLIED = False


def patch_gemma4_hybrid_mask() -> bool:
    """Install the Gemma 4 hybrid-attention mask fix.

    Returns ``True`` if the patch was installed (or was already installed),
    ``False`` if the target module could not be imported (e.g. transformers
    version predates Gemma 4) — in which case nothing is done and the
    caller can continue unaffected.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        LOG.debug(
            "gemma4_hybrid_mask: transformers.models.gemma4 not importable, "
            "skipping. This is fine for non-Gemma4 training."
        )
        return False

    if not hasattr(modeling_gemma4, "create_causal_mask"):
        LOG.warning(
            "gemma4_hybrid_mask: modeling_gemma4 has no 'create_causal_mask' "
            "binding, skipping. Transformers API may have changed."
        )
        return False

    original = modeling_gemma4.create_causal_mask

    def hybrid_create_causal_mask(config: Any, *args: Any, **kwargs: Any):
        """Wrapper that forces SDPA format for the full-attention mask.

        The global layers were patched to SDPA by
        ``_apply_gemma_hybrid_attention``, so their mask must be 4D. The
        original ``create_causal_mask`` dispatches on
        ``config._attn_implementation``; we shadow that with a local
        override.
        """
        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        return original(sdpa_config, *args, **kwargs)

    # Preserve the original reference on the wrapper for tests / teardown.
    hybrid_create_causal_mask._axolotl_original = original  # type: ignore[attr-defined]

    modeling_gemma4.create_causal_mask = hybrid_create_causal_mask
    _PATCH_APPLIED = True
    LOG.info(
        "gemma4_hybrid_mask: patched modeling_gemma4.create_causal_mask to "
        "force SDPA-format masks for full-attention layers"
    )
    return True


def unpatch_gemma4_hybrid_mask() -> None:
    """Restore the original ``create_causal_mask``. Useful for tests."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        return
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        _PATCH_APPLIED = False
        return
    current = modeling_gemma4.create_causal_mask
    original = getattr(current, "_axolotl_original", None)
    if original is not None:
        modeling_gemma4.create_causal_mask = original
    _PATCH_APPLIED = False
