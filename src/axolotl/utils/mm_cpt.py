"""Shared predicate for multimodal CPT dataset entries."""

from __future__ import annotations

from typing import Any


def is_mm_cpt_entry(entry: Any) -> bool:
    """True when a dataset entry (dict or pydantic) opts into multimodal CPT."""
    if entry is None:
        return False
    if isinstance(entry, dict):
        ds_type = entry.get("type")
        mm_flag = entry.get("multimodal")
    else:
        ds_type = getattr(entry, "type", None)
        mm_flag = getattr(entry, "multimodal", None)
    if isinstance(ds_type, str):
        ds_type = ds_type.strip()
    return ds_type == "multimodal_pretrain" or bool(mm_flag)
