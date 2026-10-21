"""Shared validation for normalized typed-decision source records."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Literal

LabelKind = Literal["hard", "dist", "set"]


def decode_json(value: Any) -> Any:
    """Decode structured source fields while retaining ordinary text states."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def probabilities(values: Sequence[float], count: int) -> list[float]:
    if len(values) != count or count < 2:
        raise ValueError("target must contain one probability per alternative")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("target probabilities must be finite and nonnegative")
    total = sum(result)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-4):
        raise ValueError("target probabilities must sum to one")
    return [value / total for value in result]


def label_target(
    values: Sequence[float],
    kind: LabelKind,
    *,
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    support = [index for index, value in enumerate(values) if value > 0]
    if kind == "dist":
        if candidate_ids is None:
            return {"kind": "dist", "probs": list(values)}
        if len(candidate_ids) != len(values):
            raise ValueError("soft candidate IDs must align with probabilities")
        return {
            "kind": "dist",
            "candidate_indices": support,
            "candidate_ids": [str(candidate_ids[index]) for index in support],
            "probs": [values[index] for index in support],
            "other_probability": sum(
                value for index, value in enumerate(values) if index not in support
            ),
        }
    if kind == "set":
        if not support:
            raise ValueError("set target must contain an allowed alternative")
        return {"kind": "set", "allowed_set": support}
    if kind != "hard":
        raise ValueError(f"unknown target kind: {kind}")
    if len(support) != 1:
        raise ValueError("hard target must be one-hot; refusing to discard soft mass")
    return {"kind": "hard", "gold_idx": support[0]}


def finite_question(
    kind: str, instructions: str, options: Sequence[str], target: Sequence[float]
) -> tuple[dict[str, Any], list[float]]:
    """Normalize a question and remap its target to the serving option order."""
    names = [str(option) for option in options]
    if len(set(names)) != len(names):
        raise ValueError("question alternatives must have distinct names")
    values = probabilities(target, len(names))
    question: dict[str, Any] = {"type": kind, "instructions": instructions}
    if kind in {"noul", "bool", "boolean"}:
        aliases = {"true": "yes", "false": "no", "yes": "yes", "no": "no"}
        canonical = [aliases.get(name.lower()) for name in names]
        if sorted(name for name in canonical if name is not None) != ["no", "yes"]:
            raise ValueError(
                "boolean alternatives must be exactly yes/no or true/false"
            )
        question["type"] = "noul"
        values = [values[canonical.index(name)] for name in ("yes", "no")]
    elif kind == "choice":
        question["options"] = names
    elif kind == "score":
        question["levels"] = names
    else:
        raise ValueError(f"unsupported finite question type: {kind}")
    return question, values
