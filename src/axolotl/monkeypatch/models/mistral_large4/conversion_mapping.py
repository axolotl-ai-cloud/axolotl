"""Make the Mistral Large 4 MoE weight renamings idempotent.

The checkpoint conversion renames ``.experts.`` / ``.shared_experts.`` / ``.gate.`` to
``.mlp.<name>.``. PEFT re-applies a model's renamings to adapter keys that are already in
HF format, turning ``.mlp.experts.`` into ``.mlp.mlp.experts.``, so the expert, shared-expert
and router LoRA silently fail to load. A ``(?<!\\.mlp)`` lookbehind makes them fixed points.
"""

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MODEL_TYPES = ("mistral_large4", "mistral_large4_text")
_MOE_MODULES = ("gate", "experts", "shared_experts")


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
    """Re-register the mistral_large4 conversion mappings with idempotent MoE renamings.

    Returns the number of renamings rewritten (0 when transformers already ships them fixed).
    """
    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        return 0

    total = 0
    for model_type in MODEL_TYPES:
        mapping = get_checkpoint_conversion_mapping(model_type)
        if not mapping:
            continue
        patched, changed = _idempotent_renamings(mapping)
        if changed:
            register_checkpoint_conversion_mapping(model_type, patched, overwrite=True)
            total += changed
    if total:
        LOG.info(
            f"Made {total} mistral_large4 MoE weight renamings idempotent for PEFT"
        )
    return total
