"""Normalize the multi-question typed-decisions evaluation corpus."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .common import decode_json, finite_question, label_target


def normalize(
    row: Mapping[str, Any], *, source_split: str | None = None
) -> dict[str, Any]:
    row_split = row.get("split")
    if row_split is not None and (
        not isinstance(row_split, str) or row_split.lower() != source_split
    ):
        raise ValueError("typed decisions row split disagrees with declared split")
    questions, gold = decode_json(row["questions"]), decode_json(row["gold"])
    if not isinstance(questions, dict) or not questions or not isinstance(gold, dict):
        raise ValueError("typed decisions require questions and gold objects")
    if set(questions) != set(gold):
        raise ValueError("each typed question must have exactly one target")
    normalized, labels = {}, {}
    for key, question in questions.items():
        answer = gold[key]
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict):
                raise ValueError("choice criteria must map names to descriptions")
            identities = list(criteria)
            names = identities
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError("score criteria must be an ordered list")
            identities = [str(index) for index in range(len(criteria))]
            names = [str(value) for value in criteria]
        elif kind == "noul":
            identities, names = ["false", "true"], ["no", "yes"]
        else:
            raise ValueError(f"unsupported typed question type: {kind}")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(identities):
            raise ValueError(
                f"question {key!r} needs probabilities for each alternative"
            )
        result, values = finite_question(
            kind,
            str(question.get("instructions", "")),
            names,
            [probs[identity] for identity in identities],
        )
        if kind == "noul":
            identities = ["yes", "no"]
        if kind == "choice":
            assert isinstance(criteria, dict)
            result["options"] = [
                {"name": name, "description": criteria[name]} for name in names
            ]
        elif kind == "noul" and criteria:
            result["criteria"] = criteria
        normalized[key] = result
        smoothing = answer.get("smoothing")
        if smoothing is None:
            labels[key] = label_target(values, "dist", candidate_ids=identities)
        else:
            if (
                isinstance(smoothing, bool)
                or not isinstance(smoothing, (float, int))
                or not math.isfinite(smoothing)
                or not 0.0 <= smoothing < 1.0
            ):
                raise ValueError(
                    f"question {key!r} hard smoothing must be finite and in [0, 1)"
                )
            target = label_target(values, "hard")
            target["smoothing"] = float(smoothing)
            labels[key] = target
    state = decode_json(row["state"])
    if state is None:
        raise ValueError("typed record requires a state")
    workflow = row["workflow"]
    if not isinstance(workflow, str) or not workflow:
        raise ValueError("typed record requires workflow metadata")
    return {
        "source": "typed_decisions",
        "family": workflow,
        "id": str(row["id"]),
        "group": str(row["id"]),
        "state": state,
        "questions": normalized,
        "labels": labels,
        "source_metadata": {
            "provenance": {
                "dataset": "LocalLLaMA/typed-decisions",
                "workflow": workflow,
                "split": source_split,
                "row_split": row_split,
                "source_id": str(row["id"]),
            }
        },
    }
