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

# Attention-interface name for Gemma-4 global (full_attention) layers. They have head_dim=512, so FA2
# can't serve them and the model (FA2 at the top level) hands them no mask, relying on cu_seqlens that
# only the FA2 sliding layers consume. With sample packing that means the global layers would attend
# ACROSS document boundaries (pure causal). This impl rebuilds the block-diagonal-causal mask from
# position_ids so the globals respect doc boundaries, on the memory-efficient SDPA backend.
GLOBAL_PACKED_SDPA = "sdpa_global_packed"


# When set, the head_dim=512 global layers use the Triton flash_d512 kernel (fwd+bwd, varlen) instead
# of the SDPA efficient backend (~2x faster at head_dim 512). Set from cfg.flash_attn_d512.
def set_flash_d512(enabled: bool) -> None:
    """Backwards-compat shim: the head_dim>256 routing is now the generic large_head_attention
    capability. True -> 'auto' (flash only on packed rows, the proven win)."""
    from axolotl.monkeypatch.attention.large_head import set_large_head_policy

    set_large_head_policy("auto" if enabled else "sdpa")


def _packing_block_causal_mask(position_ids, dtype, device):
    """Block-diagonal causal additive mask [B,1,S,S] from packed position_ids (which reset to 0 at
    each document start). -inf across document boundaries and for non-causal positions, 0 elsewhere."""
    import torch

    if position_ids.dim() == 1:
        position_ids = position_ids[None]
    Bz, Sz = position_ids.shape
    doc = (position_ids == 0).cumsum(-1)  # [B,S] document index (1-based)
    same_doc = doc[:, :, None] == doc[:, None, :]  # [B,S,S]
    causal = torch.ones(Sz, Sz, dtype=torch.bool, device=device).tril()[None]  # [1,S,S]
    allow = same_doc & causal
    mask = torch.zeros(Bz, 1, Sz, Sz, dtype=dtype, device=device)
    mask.masked_fill_(~allow[:, None], torch.finfo(dtype).min)
    return mask


def _register_global_packed_sdpa() -> None:
    """Register the packing-aware global-layer attention impl (block-diagonal mask + efficient SDPA)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if GLOBAL_PACKED_SDPA in ALL_ATTENTION_FUNCTIONS.valid_keys():
        return

    def sdpa_global_packed_forward(module, query, key, value, attention_mask, **kwargs):
        # The top-level FA2 path leaves these layers maskless and carries packing only via cu_seqlens
        # (consumed by FA2, not SDPA). Rebuild the block-diagonal mask from position_ids so the global
        # layers don't cross document boundaries. Single-document rows stay maskless (is_causal).
        from axolotl.monkeypatch.attention.large_head import flash_d512_route

        position_ids = kwargs.get("position_ids")
        # The generic large-head router takes head_dim>256 packed rows through the Triton flash
        # kernel (per the large_head_attention policy); ~2.7x the 4D-mask SDPA, ~3x less memory than
        # nested-tensor SDPA. It declines (returns None) for single-doc/policy=sdpa -> SDPA below.
        if attention_mask is None:
            routed = flash_d512_route(
                module, query, key, value, kwargs.get("scaling"), position_ids
            )
            if routed is not None:
                return routed
        # Packing detection: static under a declared config (compile-clean), runtime probe otherwise.
        pid = None
        if attention_mask is None:
            from axolotl.monkeypatch.attention.large_head import (
                _multidoc_position_ids,
            )

            pid = _multidoc_position_ids(position_ids)
        # Packed without the kernel -> block-diagonal mask so globals respect doc boundaries.
        # Single-document -> mask stays None (SDPA is_causal, the fast path).
        if pid is not None:
            attention_mask = _packing_block_causal_mask(pid, query.dtype, query.device)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    ALL_ATTENTION_FUNCTIONS.register(GLOBAL_PACKED_SDPA, sdpa_global_packed_forward)
    LOG.info(
        "gemma4_hybrid_mask: registered '%s' (block-diagonal packing mask for head_dim=512 global "
        "layers so they respect document boundaries under sample packing)",
        GLOBAL_PACKED_SDPA,
    )


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


def _patch_use_gqa_head_dim_guard() -> bool:
    """Stop ``enable_gqa`` from forcing the MATH SDPA backend on large-head-dim layers.

    ``sdpa_attention_forward`` enables ``enable_gqa=True`` whenever ``attention_mask is None``
    (``use_gqa_in_sdpa``), with no head_dim check. But SDPA's flash/efficient GQA path only
    supports head_dim <= 256; at head_dim > 256 ``enable_gqa`` silently falls back to the MATH
    kernel, which materializes the full [H, S, S] scores. Repeating KV instead keeps the
    memory-efficient backend. For Gemma-4's head_dim=512 global layers this is ~2.9 GiB -> ~0.2 GiB
    per layer with identical math (repeat_kv == GQA).
    """
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
    except ImportError:
        return False
    original = sdpa_mod.use_gqa_in_sdpa
    if getattr(original, "_axolotl_head_dim_guarded", False):
        return True

    # *args/**kwargs: transformers 5.13 adds a `value` positional arg
    def use_gqa_in_sdpa_guarded(attention_mask, key, *args, **kwargs):
        # head_dim > 256 -> enable_gqa drops to the MATH backend; force repeat_kv (efficient) instead.
        if key.shape[-1] > 256:
            return False
        return original(attention_mask, key, *args, **kwargs)

    use_gqa_in_sdpa_guarded._axolotl_head_dim_guarded = True  # type: ignore[attr-defined]
    use_gqa_in_sdpa_guarded._axolotl_original = original  # type: ignore[attr-defined]
    sdpa_mod.use_gqa_in_sdpa = use_gqa_in_sdpa_guarded
    LOG.info(
        "gemma4_hybrid_mask: guarded use_gqa_in_sdpa (head_dim>256 -> repeat_kv, not enable_gqa) "
        "to keep the memory-efficient SDPA backend on head_dim=512 global layers"
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

    if not patched_any:
        return False

    # Only touch global SDPA state once we know a Gemma4 namespace was actually patched —
    # otherwise _PATCH_APPLIED stays False and unpatch() would skip cleaning these up.
    _patch_use_gqa_head_dim_guard()
    _register_global_packed_sdpa()
    _PATCH_APPLIED = True
    return True


def unpatch_gemma4_hybrid_mask() -> None:
    """Restore the original ``create_causal_mask`` in every namespace. Tests."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        return
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod

        guarded = getattr(sdpa_mod, "use_gqa_in_sdpa", None)
        original = getattr(guarded, "_axolotl_original", None)
        if original is not None:
            sdpa_mod.use_gqa_in_sdpa = original
    except ImportError:
        pass
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
