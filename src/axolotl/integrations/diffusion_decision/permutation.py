"""Deterministic question and choice ordering with target preservation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from ._util import canonical_json
from .adapters.jsonl import normalize_jsonl


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
    value = {
        key: record.get(key) for key in ("source", "group", "id", "state", "questions")
    }
    return canonical_json(value, allow_nan=False)
