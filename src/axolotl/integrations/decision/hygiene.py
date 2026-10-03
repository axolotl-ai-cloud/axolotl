"""State decontamination and family boundaries for decision datasets."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


def state_fingerprint(state: Any) -> str:
    """Hash canonical state content independently of questions and targets."""
    if state is None:
        raise ValueError("decision record requires a state")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            pass
    if isinstance(state, str):
        text = " ".join(unicodedata.normalize("NFC", state).split())
    else:
        text = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def family_key(record: Mapping[str, Any]) -> tuple[str, str]:
    source = record.get("source")
    family = record.get("family") or record.get("group")
    if not isinstance(source, str) or not source:
        raise ValueError("decision record requires a source")
    if not isinstance(family, str) or not family:
        raise ValueError("decision record requires a family or group")
    return source, family


def decontaminate(
    training: Iterable[Mapping[str, Any]],
    evaluation: Iterable[Mapping[str, Any]],
    *,
    exclude_families: bool = True,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Remove evaluation states across sources and optionally held-out families."""
    eval_states: set[str] = set()
    eval_families: set[tuple[str, str]] = set()
    for record in evaluation:
        eval_states.add(state_fingerprint(record["state"]))
        eval_families.add(family_key(record))
    kept = []
    counts = {"input": 0, "state_overlap": 0, "family_overlap": 0, "kept": 0}
    for record in training:
        counts["input"] += 1
        key = family_key(record)
        if state_fingerprint(record["state"]) in eval_states:
            counts["state_overlap"] += 1
        elif exclude_families and key in eval_families:
            counts["family_overlap"] += 1
        else:
            kept.append(record)
            counts["kept"] += 1
    return kept, counts


def assert_split_isolation(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Reject state or source-family leakage between named dataset splits."""
    states: dict[str, str] = {}
    families: dict[tuple[str, str], str] = {}
    for split, records in splits.items():
        for record in records:
            fingerprint = state_fingerprint(record["state"])
            key = family_key(record)
            if fingerprint in states and states[fingerprint] != split:
                raise ValueError(
                    f"state overlaps splits {states[fingerprint]!r} and {split!r}"
                )
            if key in families and families[key] != split:
                raise ValueError(
                    f"family {key!r} overlaps splits {families[key]!r} and {split!r}"
                )
            states[fingerprint] = split
            families[key] = split
