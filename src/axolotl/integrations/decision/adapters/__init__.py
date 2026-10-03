"""Source adapters for typed-decision datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .autojev_route import normalize_autojev_route, validate_autojev_request
from .jev_bench import normalize_jev_bench
from .jev_distill import normalize_jev_distill
from .jsonl import normalize_jsonl
from .nimble import normalize_nimble
from .open_jev import normalize_open_jev
from .procedural import normalize as normalize_procedural
from .typed_decisions import normalize as normalize_typed_decisions

_ADAPTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "open_jev": normalize_open_jev,
    "autojev_route": normalize_autojev_route,
    "procedural": normalize_procedural,
    "jev_bench": normalize_jev_bench,
    "jev_distill": normalize_jev_distill,
    "nimble": normalize_nimble,
    "typed_decisions": normalize_typed_decisions,
    "jsonl": normalize_jsonl,
}


def normalize_record(
    adapter: str,
    row: Mapping[str, Any],
    *,
    training: bool,
    target_basis: Mapping[str, str] | None = None,
    codebook: str = "vendored26",
    source_split: str | None = None,
) -> dict[str, Any]:
    name = adapter.removeprefix("decision.")
    if name not in _ADAPTERS:
        raise ValueError(f"unknown decision adapter: {adapter}")
    if training and name == "typed_decisions" and source_split != "train":
        raise ValueError("typed_decisions training requires the declared train split")
    kwargs: dict[str, Any] = {}
    if name == "open_jev" and target_basis is not None:
        kwargs["target_basis"] = target_basis
    if name == "jsonl":
        kwargs["codebook"] = codebook
    if name == "typed_decisions":
        kwargs["source_split"] = source_split
    result = _ADAPTERS[name](row, **kwargs)
    if name == "jsonl":
        if training and result.get("source") == "typed_decisions":
            raise ValueError(
                "typed_decisions is reserved for evaluation when supplied through jsonl"
            )
        return result
    return normalize_jsonl(result, codebook=codebook)


__all__ = [
    "normalize_record",
    "normalize_open_jev",
    "normalize_autojev_route",
    "validate_autojev_request",
    "normalize_procedural",
    "normalize_jev_bench",
    "normalize_jev_distill",
    "normalize_nimble",
    "normalize_typed_decisions",
    "normalize_jsonl",
]
