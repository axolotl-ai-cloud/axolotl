"""Teacher distributions from the Jev distillation corpus."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import decode_json, finite_question, label_target


def normalize_jev_distill(row: Mapping[str, Any]) -> dict[str, Any]:
    question, target = finite_question(
        str(row["kind"]), str(row["question"]), row["options"], row["target"]
    )
    state = decode_json(row["state"])
    if state is None:
        raise ValueError("distillation record requires a state")
    family, domain = str(row["family"]), str(row["domain"])
    if not family or not domain:
        raise ValueError("distillation record requires family and domain")
    return {
        "id": str(row["id"]),
        "source": "jev_distill",
        "state": state,
        "questions": {"q1": question},
        "labels": {"q1": label_target(target, "dist")},
        "family": family,
        "group": str(row["id"]),
        "source_metadata": {"domain": domain, "source": row.get("source")},
    }
