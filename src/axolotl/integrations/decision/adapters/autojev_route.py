"""Normalize raw AutoJev Decisions API route requests for training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .common import probabilities


def validate_autojev_request(
    request: Any, *, min_candidates: int = 2
) -> Mapping[str, Any]:
    """Validate the prompt-free one-route Decisions API request contract."""
    if not isinstance(request, Mapping):
        raise ValueError("AutoJev request must be an object")
    if set(request) != {"model", "state", "questions"}:
        raise ValueError(
            "AutoJev request must contain exactly model, state, and questions"
        )
    if not isinstance(request["model"], str) or not request["model"].strip():
        raise ValueError("AutoJev request model must be a nonempty string")
    if not isinstance(request["state"], Mapping):
        raise ValueError("AutoJev request state must be an object")
    questions = request["questions"]
    if not isinstance(questions, Mapping) or set(questions) != {"route"}:
        raise ValueError("AutoJev route training accepts exactly one route question")
    route = questions["route"]
    if not isinstance(route, Mapping) or route.get("type") != "choice":
        raise ValueError("AutoJev route question must have type choice")
    if not isinstance(route.get("instructions"), str):
        raise ValueError("AutoJev route question requires string instructions")
    criteria = route.get("criteria")
    if min_candidates < 1:
        raise ValueError("min_candidates must be positive")
    if not isinstance(criteria, Mapping) or len(criteria) < min_candidates:
        raise ValueError(
            f"AutoJev route criteria must contain at least {min_candidates} candidates"
        )
    for candidate_id, description in criteria.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("AutoJev candidate IDs must be nonempty strings")
        if not isinstance(description, str):
            raise ValueError("AutoJev candidate descriptions must be JSON strings")
        try:
            metadata = json.loads(description)
        except json.JSONDecodeError as error:
            raise ValueError(
                "AutoJev candidate descriptions must be valid JSON strings"
            ) from error
        if not isinstance(metadata, Mapping):
            raise ValueError("AutoJev candidate description JSON must be an object")
    return request


def _label(row: Mapping[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    has_gold = "gold_unique_model_id" in row
    has_probabilities = "unique_model_id_probabilities" in row
    if has_gold == has_probabilities:
        raise ValueError(
            "AutoJev route record requires exactly one of gold_unique_model_id "
            "or unique_model_id_probabilities"
        )
    if has_gold:
        candidate_id = row["gold_unique_model_id"]
        if not isinstance(candidate_id, str) or candidate_id not in candidate_ids:
            raise ValueError("AutoJev gold unique model ID must be a listed candidate")
        return {"kind": "hard", "gold_idx": candidate_ids.index(candidate_id)}

    values_by_id = row["unique_model_id_probabilities"]
    if not isinstance(values_by_id, Mapping) or set(values_by_id) != set(candidate_ids):
        raise ValueError(
            "AutoJev soft probabilities must specify every listed candidate ID"
        )
    return {
        "kind": "dist",
        "probs": probabilities(
            [values_by_id[candidate_id] for candidate_id in candidate_ids],
            len(candidate_ids),
        ),
    }


def normalize_autojev_route(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one captured AutoJev Decisions request without changing its data."""
    request = validate_autojev_request(row.get("request"))
    record_id = row.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("AutoJev route record requires a nonempty id")
    request_id = (
        "autojev:"
        + hashlib.sha256(
            json.dumps(
                request, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    supplied_request_id = row.get("request_id")
    if supplied_request_id is not None and supplied_request_id != request_id:
        raise ValueError(
            "AutoJev route request_id does not match canonical request hash"
        )
    group = row.get("group", record_id)
    if not isinstance(group, str) or not group:
        raise ValueError("AutoJev route group must be a nonempty string")
    criteria = request["questions"]["route"]["criteria"]
    assert isinstance(criteria, Mapping)
    candidate_ids = list(criteria)
    metadata = row.get("source_metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("AutoJev source_metadata must be an object when supplied")
    source_metadata = deepcopy(dict(metadata))
    provenance = source_metadata.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("AutoJev source_metadata.provenance must be an object")
    source_metadata["provenance"] = {
        **deepcopy(dict(provenance)),
        "adapter": "autojev_route",
        "source_id": record_id,
        "decision_model": request["model"],
    }
    source_metadata["autojev_request"] = deepcopy(dict(request))
    return {
        "id": record_id,
        "request_id": request_id,
        "source": "autojev_route",
        "group": group,
        "family": "autojev_route",
        "state": deepcopy(request["state"]),
        "questions": {
            "route": {
                "type": "choice",
                "instructions": request["questions"]["route"]["instructions"],
                "options": [
                    {"name": candidate_id, "description": criteria[candidate_id]}
                    for candidate_id in candidate_ids
                ],
            }
        },
        "labels": {"route": _label(row, candidate_ids)},
        "source_metadata": source_metadata,
    }
