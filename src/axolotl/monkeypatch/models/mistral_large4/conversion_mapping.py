"""Make the Mistral Large 4 MoE weight renamings idempotent.

The checkpoint conversion renames ``.experts.`` / ``.shared_experts.`` / ``.gate.`` to
``.mlp.<name>.``. PEFT re-applies a model's renamings to adapter keys that are already in
HF format, turning ``.mlp.experts.`` into ``.mlp.mlp.experts.``, so the expert, shared-expert
and router LoRA silently fail to load. A ``(?<!\\.mlp)`` lookbehind makes them fixed points.
"""

import hashlib
import json

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MODEL_TYPE = "mistral_large4_text"
_MOE_MODULES = ("gate", "experts", "shared_experts")
# sha256 of the mistral_large4_text renaming list: as shipped in the transformers PR, and
# with the lookbehinds already applied (by this patch, or upstream).
_UNPATCHED_SHA256 = "2f191679f0ac1402d6d6ecd3d16527b8f495d8695a929d0f560fdf3a95551b0e"
_PATCHED_SHA256 = "c6b2951b579b4b80bb390746590d315d7c9ec0de39d8c1d0e47070822c4ee53b"


def _renamings_fingerprint(mapping) -> str:
    from transformers.core_model_loading import WeightRenaming

    renamings = [
        [t._original_source_patterns, t._original_target_patterns]
        for t in mapping
        if type(t) is WeightRenaming
    ]
    return hashlib.sha256(json.dumps(renamings).encode()).hexdigest()


def _idempotent_renamings(mapping):
    from transformers.core_model_loading import WeightRenaming

    patched, changed = [], 0
    for transform in mapping:
        source = getattr(transform, "_original_source_patterns", None)
        if type(transform) is WeightRenaming and source and len(source) == 1:
            for name in _MOE_MODULES:
                if source[0] == rf"\.{name}\.":
                    transform = WeightRenaming(
                        rf"(?<!\.mlp)\.{name}\.",
                        transform._original_target_patterns[0],
                    )
                    changed += 1
        patched.append(transform)
    return patched, changed


def patch_mistral_large4_moe_renamings() -> int:
    """Re-register the mistral_large4_text conversion mapping with idempotent MoE renamings.

    Returns the number of renamings rewritten: 0 when transformers lacks the model or already
    ships the fix. Raises if the upstream renamings changed, since the patch was written
    against a specific version of them.
    """
    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        return 0

    mapping = get_checkpoint_conversion_mapping(MODEL_TYPE)
    if not mapping:
        return 0
    fingerprint = _renamings_fingerprint(mapping)
    if fingerprint == _PATCHED_SHA256:
        return 0
    if fingerprint != _UNPATCHED_SHA256:
        raise ValueError(
            f"transformers' {MODEL_TYPE} weight renamings changed (sha256 {fingerprint}), so "
            "axolotl's idempotent MoE renaming patch is not valid anymore. Please inform the "
            "axolotl maintainers."
        )
    patched, changed = _idempotent_renamings(mapping)
    register_checkpoint_conversion_mapping(MODEL_TYPE, patched, overwrite=True)
    LOG.info(f"Made {changed} {MODEL_TYPE} MoE weight renamings idempotent for PEFT")
    return changed
