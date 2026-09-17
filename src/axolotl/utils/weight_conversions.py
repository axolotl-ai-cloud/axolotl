"""Shared helper for registering transformers checkpoint weight conversions.

``register_checkpoint_conversion_mapping`` replaces the entire converter list for a
``model_type`` rather than merging, and ``rename_source_key`` picks the FIRST converter
whose (unanchored) source pattern matches a key. Registering under a stock ``model_type``
therefore has to be deliberate about whether to keep the built-ins transformers ships and
about ordering, since an earlier converter shadows a later one for the same key.
"""

from __future__ import annotations

from collections.abc import Sequence


def _source_key(entry) -> tuple:
    return tuple(getattr(entry, "source_patterns", ()) or ())


def merge_weight_conversions(model_type: str, entries: Sequence) -> list:
    """``entries`` first, then the existing converters they don't supersede.

    New entries lead so they win ``rename_source_key``'s first-match; the remaining
    built-ins act as fallbacks for keys the new entries don't claim. An existing entry
    with the same source-pattern set as an incoming one is dropped, which also makes
    re-registration idempotent.

    Caveat: first-match ordering is what ``WeightConverter``s need, but ``WeightRenaming``s
    all fire and chain in list order (and the save-side reverse depends on that order), so a
    profile renaming placed ahead of a stock type's built-in renamings could chain into the
    wrong keys. No shipped profile registers renamings under a stock model_type today.
    """
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    entries = list(entries)
    existing = get_checkpoint_conversion_mapping(model_type) or []
    incoming = {_source_key(entry) for entry in entries}
    fallback = [entry for entry in existing if _source_key(entry) not in incoming]
    return entries + fallback


_MERGE_REGISTERED_MODEL_TYPES: set[str] = set()


def register_weight_conversions(
    model_type: str, entries: Sequence, *, replace_existing: bool = False
) -> None:
    """Register ``entries`` for ``model_type`` (always with ``overwrite=True``).

    ``replace_existing=False`` merges with whatever is already registered
    (:func:`merge_weight_conversions`), keeping built-ins as fallbacks. Set
    ``replace_existing=True`` when the new converters must be the ONLY ones able to claim
    their keys, e.g. a quantized-expert loader whose fast path relies on the stock fusion
    converters being absent.

    A ``replace_existing`` call raises ``NotImplementedError`` if a merge-mode registration
    already ran for the same ``model_type``: stacking a model-support profile's conversions
    with a replace-mode consumer (e.g. the NVFP4 expert converters) would silently drop the
    profile's, which is not supported yet.
    """
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    if replace_existing:
        if model_type in _MERGE_REGISTERED_MODEL_TYPES:
            raise NotImplementedError(
                f"Cannot register replace-mode weight conversions for model_type "
                f"'{model_type}': a model-support profile already registered conversions for it, "
                f"and replacing would silently drop them. Stacking a profile with a replace-mode "
                f"consumer (e.g. NVFP4 expert converters) under one model_type is not supported "
                f"yet; please open an issue at https://github.com/axolotl-ai-cloud/axolotl/issues."
            )
        payload = list(entries)
    else:
        _MERGE_REGISTERED_MODEL_TYPES.add(model_type)
        payload = merge_weight_conversions(model_type, entries)
    register_checkpoint_conversion_mapping(model_type, payload, overwrite=True)
