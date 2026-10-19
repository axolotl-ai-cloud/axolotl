"""Deterministic question and choice ordering with target preservation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from ._util import canonical_json
from .adapters.jsonl import normalize_jsonl
from .template import codebook_labels


def draw_stream(draw: tuple[int, int] | None) -> str:
    """Stream suffix for one draw of a record: (epoch, draw number within the epoch).

    The first draw of a record in epoch 0 keeps the unsuffixed streams, so its order
    matches records permuted before draws were distinguished (and the prepared canvas).
    """
    if draw is None:
        return ""
    epoch, number = draw
    if (epoch, number) == (0, 0):
        return ""
    if min(epoch, number) < 0:
        raise ValueError("decision draw epoch and number must be nonnegative")
    return f"@{int(epoch)}.{int(number)}"


def permute_record(
    record: Mapping[str, Any],
    *,
    seed: int,
    codebook: str = "vendored26",
    draw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Shuffle question and choice order, seeded by the record's identity and draw.

    ``draw`` is (epoch, draw number of this record within the epoch); every draw
    but (0, 0) gets its own order.
    """
    return permute_record_with_orders(record, seed=seed, codebook=codebook, draw=draw)[
        0
    ]


def permute_record_with_orders(
    record: Mapping[str, Any],
    *,
    seed: int,
    codebook: str = "vendored26",
    draw: tuple[int, int] | None = None,
    share: bool = False,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """:func:`permute_record`, plus each choice question's option order.

    ``orders[key][new_position]`` is the option's index in the normalized
    (source-ordered) record, which is how pre-tokenized segments are keyed.
    ``share=True`` returns an equal record that shares the values this function
    does not change (state, metadata, options, non-choice questions) with
    ``record`` instead of deep-copying them; callers must not mutate them.
    """
    result = normalize_jsonl(record, codebook=codebook, share=share)
    identity = _permutation_identity(result)
    suffix = draw_stream(draw)
    keys = sorted(result["questions"])
    _generator(seed, identity, f"questions{suffix}").shuffle(keys)
    result["questions"] = {key: result["questions"][key] for key in keys}
    result["labels"] = {key: result["labels"][key] for key in keys}
    orders: dict[str, list[int]] = {}
    for key, question in result["questions"].items():
        if question["type"] != "choice":
            continue
        if share:
            question = result["questions"][key] = dict(question)
        options = question["options"]
        order = list(range(len(options)))
        _generator(seed, identity, f"choices:{key}{suffix}").shuffle(order)
        orders[key] = order
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
    return result, orders


def shuffle_labels(
    record: Mapping[str, Any],
    *,
    seed: int,
    probability: float,
    codebook: str = "vendored26",
    draw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """With ``probability`` per choice question, give its options a random subset of the codebook (in codebook
    order) through ``label_indices`` instead of the first N labels. Deterministic per seed, record and question;
    returns the record unchanged (not copied) when nothing is drawn."""
    if probability <= 0.0:
        return dict(record)
    size = len(codebook_labels(codebook))
    identity = _permutation_identity(record)
    suffix = draw_stream(draw)
    result = None
    for key, question in record["questions"].items():
        if question.get("type") != "choice" or "label_indices" in question:
            continue
        count = len(question.get("options") or ())
        if count >= size:
            continue
        generator = _generator(seed, identity, f"labels:{key}{suffix}")
        if generator.random() >= probability:
            continue
        if result is None:
            result = dict(record)
            result["questions"] = dict(record["questions"])
        result["questions"][key] = {
            **question,
            "label_indices": sorted(generator.sample(range(size), count)),
        }
    return dict(record) if result is None else result


def _generator(seed: int, identity: str, stream: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stream}:{identity}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))  # nosec B311 - Dataset ordering.


def _permutation_identity(record: Mapping[str, Any]) -> str:
    value = {
        key: record.get(key) for key in ("source", "group", "id", "state", "questions")
    }
    return canonical_json(value, allow_nan=False)
