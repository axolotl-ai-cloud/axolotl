"""Validate user-supplied records in the normalized decision format."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..template import parse_decision_schema
from .common import probabilities


def normalize_jsonl(
    row: Mapping[str, Any], *, codebook: str = "vendored26"
) -> dict[str, Any]:
    result = deepcopy(dict(row))
    for field in ("source", "group"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"normalized record requires {field}")
    if result.get("state") is None:
        raise ValueError("normalized record requires state")
    images = result.get("images")
    if images is not None and (
        not isinstance(images, list)
        or any(not isinstance(image, str) or not image for image in images)
    ):
        raise ValueError("images must be a list of nonempty path or URL strings")
    questions, labels = result.get("questions"), result.get("labels")
    if not isinstance(questions, dict) or not isinstance(labels, dict):
        raise ValueError("questions and labels must be objects")
    if set(questions) != set(labels):
        raise ValueError("each question must have exactly one target")
    parsed = parse_decision_schema(
        {"questions": [{**q, "id": key} for key, q in questions.items()]},
        codebook=codebook,
    )
    for question in parsed["questions"]:
        key = question["id"]
        if question["type"] not in {"noul", "choice", "score"}:
            raise ValueError("decision training requires finite label alternatives")
        count = len(question["labels"])
        target = labels[key]
        if not isinstance(target, dict):
            raise ValueError(f"question {key!r} target must be an object")
        kind = target.get("kind")
        if kind == "dist":
            indices = target.get("candidate_indices")
            if indices is None:
                target["probs"] = probabilities(target["probs"], count)
            else:
                if (
                    not isinstance(indices, list)
                    or not indices
                    or len(indices) != len(target.get("probs", ()))
                    or len(set(indices)) != len(indices)
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < count
                        for index in indices
                    )
                ):
                    raise ValueError(
                        f"question {key!r} sparse candidates must identify alternatives"
                    )
                values = [float(value) for value in target["probs"]]
                other = float(target.get("other_probability", 0.0))
                if (
                    any(not math.isfinite(value) or value < 0 for value in values)
                    or not math.isfinite(other)
                    or other < 0
                    or not math.isclose(sum(values) + other, 1.0, abs_tol=1e-4)
                ):
                    raise ValueError(
                        f"question {key!r} sparse probabilities plus "
                        "other_probability must sum to one"
                    )
                target["probs"] = values
                target["other_probability"] = other
                candidate_ids = target.get("candidate_ids")
                if candidate_ids is not None and (
                    not isinstance(candidate_ids, list)
                    or len(candidate_ids) != len(indices)
                    or any(
                        not isinstance(value, str) or not value
                        for value in candidate_ids
                    )
                ):
                    raise ValueError(
                        f"question {key!r} sparse candidate_ids must align "
                        "with candidates"
                    )
        elif kind in {"hard", "set"}:
            indices = (
                [target.get("gold_idx")]
                if kind == "hard"
                else target.get("allowed_set")
            )
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < count
                    for index in indices
                )
            ):
                raise ValueError(
                    f"question {key!r} target indices must identify alternatives"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(f"question {key!r} allowed set must be unique")
        else:
            raise ValueError(f"unknown label kind {kind!r}")
    return result
