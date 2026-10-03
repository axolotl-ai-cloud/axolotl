"""Normalize Nimble's audited single-question records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .jev_bench import normalize_jev_bench


def normalize_nimble(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row["input"]
    questions = payload["questions"]
    if not isinstance(questions, dict) or len(questions) != 1:
        raise ValueError("Nimble reference.target requires exactly one question")
    question_id, question = next(iter(questions.items()))
    target = row["reference"]["target"]
    if question["type"] == "noul":
        canonical = str(target).lower()
        aliases = {"true": "1", "false": "0", "yes": "1", "no": "0", "1": "1", "0": "0"}
        if canonical not in aliases:
            raise ValueError("unknown Nimble boolean target")
        target = aliases[canonical]
    elif question["type"] == "score":
        criteria = question["criteria"]
        value = str(target)
        if not value.isdecimal():
            names = [str(item) for item in criteria]
            if value not in names:
                raise ValueError("unknown Nimble score target")
            target = str(names.index(value))
    family = row.get("source_family") or row.get("family")
    if not isinstance(family, str) or not family:
        raise ValueError("Nimble record requires family metadata")
    result = normalize_jev_bench(
        {
            "id": row["id"],
            "source": family,
            "state": payload["state"],
            "question": question,
            "label": target,
            "soft_label": None,
            "meta": {
                key: row[key]
                for key in ("domain", "family", "method", "provenance", "variant")
                if key in row
            },
        }
    )
    if "instructions" in payload:
        result["instructions"] = payload["instructions"]
    result["source"] = "nimble"
    result["questions"] = {str(question_id): result["questions"]["q1"]}
    result["labels"] = {str(question_id): result["labels"]["q1"]}
    return result
