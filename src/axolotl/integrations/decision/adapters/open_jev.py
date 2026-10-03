"""Normalize Open-Jev records without collapsing set or distribution targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .common import LabelKind, decode_json, finite_question, label_target

_DIST_BASES = {"defined_uniform_latent_worlds", "published_human_reviewed_label"}
_SET_BASES = {"exact minimax; ties are uniform optimal actions, not win probabilities"}
_SNAKE_HEURISTIC_BASIS = (
    "Choice: visible-state BFS/flood-fill heuristic, not optimal. "
    "Noul: exact one-step collision rule."
)


def _uniform_support(values: Sequence[float]) -> bool:
    support = [value for value in values if value > 0]
    return len(support) > 1 and all(
        math.isclose(value, support[0], rel_tol=0, abs_tol=1e-12)
        for value in support[1:]
    )


def normalize_open_jev(
    row: Mapping[str, Any],
    *,
    target_basis: Mapping[str, LabelKind] | None = None,
) -> dict[str, Any]:
    metadata = decode_json(row.get("metadata_json", row.get("metadata", {})))
    if not isinstance(metadata, dict):
        raise ValueError("Open-Jev metadata must be an object")
    state = decode_json(row.get("state_json", row.get("state")))
    if state is None:
        raise ValueError("Open-Jev record needs a state")
    source = str(row["source"])
    basis = metadata.get("target_basis")
    question, values = finite_question(
        row["kind"], str(row["question"]), row["options"], row["target"]
    )
    override = (target_basis or {}).get(basis) if isinstance(basis, str) else None
    if override is not None:
        kind = override
    elif basis in _DIST_BASES:
        kind = "dist"
    elif basis in _SET_BASES or (
        source == "ir-control-v1" and metadata.get("method") == "setwise"
    ):
        kind = "set"
    elif basis == _SNAKE_HEURISTIC_BASIS and row["kind"] == "choice":
        kind = "set"
    elif (
        source == "ir-control-v1"
        and metadata.get("method") == "pairwise"
        and _uniform_support(values)
    ):
        kind = "set"
    elif sum(value > 0 for value in values) == 1:
        kind = "hard"
    else:
        raise ValueError(
            f"Open-Jev soft target needs an explicit target_basis mapping: {basis!r}"
        )
    record_id = str(row["id"])
    group = row.get("group_id")
    if not group:
        raise ValueError("Open-Jev record needs group_id for split hygiene")
    scenario_family = metadata.get("scenario_family")
    if scenario_family is not None and (
        not isinstance(scenario_family, str) or not scenario_family
    ):
        raise ValueError("Open-Jev scenario_family must be a nonempty string")
    return {
        "id": record_id,
        "source": "open_jev.wanli"
        if source == "wanli-decisions-v1"
        else "open_jev.authored",
        "state": state,
        "questions": {"q1": question},
        "labels": {"q1": label_target(values, kind)},
        "group": str(group),
        "family": scenario_family or str(group),
        "source_metadata": metadata,
    }
