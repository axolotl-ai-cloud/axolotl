"""Source adapters for typed-decision datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .jsonl import normalize_jsonl
from .procedural import normalize as normalize_procedural
from .typed_decisions import normalize as normalize_typed_decisions

_Adapter = Callable[[Mapping[str, Any], str | None], Mapping[str, Any]]

_ADAPTERS: dict[str, _Adapter] = {
    "jsonl": lambda row, _split: row,
    "procedural": lambda row, _split: normalize_procedural(row),
    "typed_decisions": lambda row, split: normalize_typed_decisions(
        row, source_split=split
    ),
}


def normalize_record(
    adapter: str,
    row: Mapping[str, Any],
    *,
    training: bool,
    codebook: str = "vendored26",
    source_split: str | None = None,
) -> dict[str, Any]:
    name = adapter.removeprefix("diffusion_decision.")
    if name not in _ADAPTERS:
        raise ValueError(f"unknown decision adapter: {adapter}")
    if training and name == "typed_decisions" and source_split != "train":
        raise ValueError("typed_decisions training requires the declared train split")
    result = normalize_jsonl(_ADAPTERS[name](row, source_split), codebook=codebook)
    if training and name == "jsonl" and result.get("source") == "typed_decisions":
        raise ValueError(
            "typed_decisions is reserved for evaluation when supplied through jsonl"
        )
    return result


__all__ = [
    "normalize_record",
    "normalize_procedural",
    "normalize_typed_decisions",
    "normalize_jsonl",
]
