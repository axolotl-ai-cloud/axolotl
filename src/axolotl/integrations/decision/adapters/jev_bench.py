"""Human hard labels and annotator distributions from JevBench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .common import LabelKind, decode_json, finite_question, label_target


def normalize_jev_bench(row: Mapping[str, Any]) -> dict[str, Any]:
    question = decode_json(row["question"])
    metadata = decode_json(row.get("meta", {}))
    if not isinstance(question, dict) or not isinstance(metadata, dict):
        raise ValueError("JevBench question and metadata must be objects")
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must map names to descriptions")
        identities = list(criteria)
        names = identities
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be an ordered list")
        identities = [str(index) for index in range(len(criteria))]
        names = [str(item) for item in criteria]
    elif kind == "noul":
        identities, names = ["0", "1"], ["no", "yes"]
    else:
        raise ValueError(f"unsupported JevBench question type: {kind}")
    soft = decode_json(row.get("soft_label"))
    target_kind: LabelKind
    if soft is None:
        gold = str(row["label"])
        if gold not in identities:
            raise ValueError(f"unknown JevBench gold label: {gold!r}")
        values = [float(identity == gold) for identity in identities]
        target_kind = "hard"
    else:
        if isinstance(soft, Mapping):
            if set(soft) != set(identities):
                raise ValueError(
                    "soft labels must specify each alternative by identity"
                )
            values = [soft[identity] for identity in identities]
        elif (
            kind == "noul"
            and isinstance(soft, (int, float))
            and not isinstance(soft, bool)
        ):
            probability_yes = float(soft)
            values = [1.0 - probability_yes, probability_yes]
        elif (
            kind == "score"
            and isinstance(soft, Sequence)
            and not isinstance(soft, (str, bytes))
        ):
            values = list(soft)
            if any(isinstance(value, bool) for value in values):
                raise ValueError("target probabilities must be finite and nonnegative")
        else:
            raise ValueError("soft labels must specify each alternative by identity")
        target_kind = "dist"
    normalized, values = finite_question(
        kind, str(question.get("instructions", "")), names, values
    )
    if kind == "choice":
        assert isinstance(criteria, dict)
        normalized["options"] = [
            {"name": name, "description": criteria[name]} for name in names
        ]
    elif kind == "noul" and criteria:
        normalized["criteria"] = criteria
    state = decode_json(row["state"])
    if state is None:
        raise ValueError("JevBench record requires a state")
    return {
        "id": str(row["id"]),
        "source": "jev_bench",
        "state": state,
        "questions": {"q1": normalized},
        "labels": {"q1": label_target(values, target_kind)},
        "family": str(row["source"]),
        "group": str(row["id"]),
        "source_metadata": metadata,
    }
