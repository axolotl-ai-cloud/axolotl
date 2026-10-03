"""Deterministic question and choice ordering with target preservation."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from typing import Any

from .adapters.jsonl import normalize_jsonl
from .template import parse_decision_schema


def permute_label_free_record(
    record: Mapping[str, Any], *, seed: int, codebook: str = "vendored26"
) -> dict[str, Any]:
    """Apply the training ordering policy to a read-only decision record."""
    if "labels" in record:
        raise ValueError("label-free permutation does not accept labels")
    result = {key: value for key, value in record.items()}
    questions = result.get("questions")
    if not isinstance(questions, Mapping) or result.get("state") is None:
        raise ValueError("label-free permutation requires state and questions")
    schema = parse_decision_schema(
        {"questions": [{**question, "id": key} for key, question in questions.items()]},
        codebook=codebook,
    )
    identity = _permutation_identity(result)
    keys = sorted(result["questions"])
    _generator(seed, identity, "questions").shuffle(keys)
    result["questions"] = {key: dict(result["questions"][key]) for key in keys}
    parsed_by_id = {question["id"]: question for question in schema["questions"]}
    for key, question in result["questions"].items():
        if parsed_by_id[key]["type"] != "choice":
            continue
        options = question.get("options")
        if not isinstance(options, list):
            raise ValueError("label-free choice question requires options")
        order = list(range(len(options)))
        _generator(seed, identity, f"choices:{key}").shuffle(order)
        question["options"] = [options[index] for index in order]
    return result


AUTOJEV_ROUTE_PERMUTATION_PROTOCOL = "request-only-v1"


def permute_record(
    record: Mapping[str, Any], *, seed: int, codebook: str = "vendored26"
) -> dict[str, Any]:
    result = normalize_jsonl(record, codebook=codebook)
    identity = _permutation_identity(result)
    keys = sorted(result["questions"])
    _generator(seed, identity, "questions").shuffle(keys)
    result["questions"] = {key: result["questions"][key] for key in keys}
    result["labels"] = {key: result["labels"][key] for key in keys}
    for key, question in result["questions"].items():
        if question["type"] != "choice":
            continue
        options = question["options"]
        order = list(range(len(options)))
        _generator(seed, identity, f"choices:{key}").shuffle(order)
        question["options"] = [options[index] for index in order]
        target = result["labels"][key]
        inverse = {old: new for new, old in enumerate(order)}
        if target["kind"] == "hard":
            target["gold_idx"] = inverse[target["gold_idx"]]
        elif target["kind"] == "dist":
            if "candidate_indices" in target:
                candidates = sorted(
                    zip(
                        (inverse[index] for index in target["candidate_indices"]),
                        target["probs"],
                        target.get("candidate_ids") or [None] * len(target["probs"]),
                        strict=True,
                    )
                )
                target["candidate_indices"] = [index for index, _, _ in candidates]
                target["probs"] = [probability for _, probability, _ in candidates]
                if target.get("candidate_ids") is not None:
                    target["candidate_ids"] = [
                        identity for _, _, identity in candidates
                    ]
            else:
                target["probs"] = [target["probs"][index] for index in order]
        else:
            target["allowed_set"] = sorted(
                inverse[index] for index in target["allowed_set"]
            )
    return result


def _generator(seed: int, identity: str, stream: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stream}:{identity}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))  # nosec B311 - Dataset ordering.


def _permutation_identity(record: Mapping[str, Any]) -> str:
    if record.get("source") == "autojev_route":
        request_id = record.get("request_id", record.get("id"))
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("AutoJev route permutation requires a request_id")
        value = {
            "source": record.get("source"),
            "id": request_id,
            "state": record.get("state"),
            "questions": record.get("questions"),
        }
    else:
        value = {
            key: record.get(key)
            for key in ("source", "group", "id", "state", "questions")
        }
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
