"""Hybrid attention mask fix for Gemma 4 (standard and unified).

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
the model's *module namespace* — NOT the original in ``masking_utils``. The
wrapper forces ``_attn_implementation="sdpa"`` on a shallow-copied config
before calling through, so the ``full_attention`` mask built inside the
text backbone's ``forward`` is always 4D/SDPA-compatible.
``create_sliding_window_causal_mask`` is left alone, so sliding-window
layers continue to receive FA2-format masks.

``gemma4_unified`` reproduces the same mixed sliding/global architecture
(``global_head_dim=512``) in its own ``modeling_gemma4_unified`` namespace,
so both namespaces are patched when present.

The patch is idempotent. Install once per process, before any Gemma 4
forward pass runs.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_APPLIED = False

# Each Gemma 4 variant fully redefines ``create_causal_mask`` in its own module
# namespace (gemma4_unified does NOT modular-import from gemma4), so both must be
# patched independently.
_TARGET_MODULES = (
    "transformers.models.gemma4.modeling_gemma4",
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
)


def _patch_module_create_causal_mask(module: Any) -> bool:
    """Wrap ``create_causal_mask`` in a single module namespace.

    Re-entry is prevented by the module-level ``_PATCH_APPLIED`` flag in
    :func:`patch_gemma4_hybrid_mask`, so this does not guard per-module.
    Returns ``True`` if patched, ``False`` if the namespace has no
    ``create_causal_mask`` binding.
    """
    if not hasattr(module, "create_causal_mask"):
        LOG.warning(
            "gemma4_hybrid_mask: %s has no 'create_causal_mask' binding, "
            "skipping. Transformers API may have changed.",
            module.__name__,
        )
        return False

    original = module.create_causal_mask

    def hybrid_create_causal_mask(config: Any, *args: Any, **kwargs: Any):
        """Force SDPA format for the full-attention mask.

        The global layers were patched to SDPA by
        ``_apply_gemma_hybrid_attention``, so their mask must be 4D. The
        original ``create_causal_mask`` dispatches on
        ``config._attn_implementation``; we shadow that with a local override
        on a shallow copy so the caller's config is left intact (the
        sliding-window factory still reads FA2 from it).
        """
        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        return original(sdpa_config, *args, **kwargs)

    # Preserve the original reference on the wrapper for tests / teardown.
    hybrid_create_causal_mask._axolotl_original = original  # type: ignore[attr-defined]
    module.create_causal_mask = hybrid_create_causal_mask
    LOG.info(
        "gemma4_hybrid_mask: patched %s.create_causal_mask to force SDPA-format "
        "masks for full-attention layers",
        module.__name__,
    )
    return True


def patch_gemma4_hybrid_mask() -> bool:
    """Install the Gemma 4 hybrid-attention mask fix across all variants.

    Returns ``True`` if at least one namespace was patched, ``False`` if none
    of the target modules could be imported (e.g. transformers version predates
    Gemma 4) — in which case nothing is done and the caller can continue
    unaffected.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    patched_any = False
    for module_path in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            LOG.debug(
                "gemma4_hybrid_mask: %s not importable, skipping. This is fine "
                "for non-Gemma4 training.",
                module_path,
            )
            continue
        if _patch_module_create_causal_mask(module):
            patched_any = True

    if patched_any:
        _PATCH_APPLIED = True
    return patched_any


def unpatch_gemma4_hybrid_mask() -> None:
    """Restore the original ``create_causal_mask`` in every namespace. Tests."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        return
    for module_path in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        current = getattr(module, "create_causal_mask", None)
        original = getattr(current, "_axolotl_original", None)
        if original is not None:
            module.create_causal_mask = original
    _PATCH_APPLIED = False
