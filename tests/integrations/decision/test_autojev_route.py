"""AutoJev route captures retain source identities and target semantics."""

import hashlib
import json

import pytest

from axolotl.integrations.decision.adapters import (
    normalize_autojev_route,
    normalize_record,
)
from axolotl.integrations.decision.permutation import (
    AUTOJEV_ROUTE_PERMUTATION_PROTOCOL,
    permute_record,
)
from axolotl.integrations.decision.preprocessing import build_decision_canvas


def request_id(request):
    return (
        "autojev:"
        + hashlib.sha256(
            json.dumps(
                request, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def record(**updates):
    result = {
        "id": "route-capture-001",
        "request": {
            "model": "~typesafe/jev-latest",
            "state": {
                "candidates": ["metadata remains in criteria"],
                "selection_policy": "cost then quality",
            },
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose exactly one candidate.",
                    "criteria": {
                        "local/model-a": '{"id":"local/model-a","cost":0.1}',
                        "cloud/model-b": '{"id":"cloud/model-b","cost":1.2}',
                        "cloud/model-c": '{"id":"cloud/model-c","cost":0.8}',
                    },
                }
            },
        },
        "gold_unique_model_id": "cloud/model-b",
        "source_metadata": {
            "provenance": {"dataset": "captured-routes", "revision": "r1"}
        },
    }
    result.update(updates)
    result.setdefault("request_id", request_id(result["request"]))
    return result


def test_hard_choice_preserves_autojev_request_data_and_identity():
    source = record()
    source.pop("request_id")
    actual = normalize_autojev_route(source)

    assert actual["state"] == source["request"]["state"]
    assert (
        actual["questions"]["route"]["instructions"] == "Choose exactly one candidate."
    )
    assert actual["questions"]["route"]["options"] == [
        {"name": candidate_id, "description": description}
        for candidate_id, description in source["request"]["questions"]["route"][
            "criteria"
        ].items()
    ]
    assert actual["labels"]["route"] == {"kind": "hard", "gold_idx": 1}
    assert actual["request_id"] == request_id(source["request"])
    assert actual["source_metadata"]["provenance"] == {
        "dataset": "captured-routes",
        "revision": "r1",
        "adapter": "autojev_route",
        "source_id": "route-capture-001",
        "decision_model": "~typesafe/jev-latest",
    }
    assert actual["source_metadata"]["autojev_request"] == source["request"]
    assert actual["group"] == source["id"]
    expected = record()
    expected.pop("request_id")
    assert source == expected


def test_multiple_captures_of_one_task_retain_the_explicit_group():
    task_group = "task:customer-support-42"
    first = normalize_autojev_route(
        record(id="autojev-observation:first", group=task_group)
    )
    second = normalize_autojev_route(
        record(id="autojev-observation:second", group=task_group)
    )

    assert first["id"] != second["id"]
    assert first["request_id"] == second["request_id"]
    assert first["group"] == second["group"] == task_group


def test_full_soft_map_and_permutation_follow_candidate_ids():
    source = record(
        unique_model_id_probabilities={
            "local/model-a": 0.1,
            "cloud/model-b": 0.7,
            "cloud/model-c": 0.2,
        }
    )
    source.pop("gold_unique_model_id")
    normalized = normalize_record("autojev_route", source, training=True)
    assert normalized["labels"]["route"] == {"kind": "dist", "probs": [0.1, 0.7, 0.2]}

    permuted = permute_record(normalized, seed=8)
    names = [option["name"] for option in permuted["questions"]["route"]["options"]]
    assert dict(zip(names, permuted["labels"]["route"]["probs"], strict=True)) == {
        "local/model-a": 0.1,
        "cloud/model-b": 0.7,
        "cloud/model-c": 0.2,
    }


def test_capture_and_inference_share_request_order_and_canvas():
    captured = normalize_autojev_route(
        record(
            id="autojev-observation:captured-once",
        )
    )
    served = {
        **captured,
        "id": captured["request_id"],
        "group": captured["request_id"],
    }

    trained = permute_record(captured, seed=23)
    inferred = permute_record(served, seed=23)

    assert AUTOJEV_ROUTE_PERMUTATION_PROTOCOL == "request-only-v1"
    assert trained["questions"] == inferred["questions"]

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return [ord(character) for character in text]

    kwargs = {
        "scaffold_ids": (1,),
        "turn_close_id": 2,
        "pad_id": 0,
        "vocab_size": 512,
        "width": 512,
        "seed": 23,
    }
    assert build_decision_canvas(
        Tokenizer(), trained, (3,), **kwargs
    ) == build_decision_canvas(Tokenizer(), inferred, (3,), **kwargs)


@pytest.mark.parametrize(
    "updates, message",
    [
        (
            {"request": {"model": "jev", "state": {}, "questions": {}}},
            "exactly one route",
        ),
        (
            {
                "request": {
                    "model": "jev",
                    "state": {},
                    "questions": {"route": {"type": "score"}},
                }
            },
            "type choice",
        ),
        (
            {
                "request": {
                    "model": "jev",
                    "state": {},
                    "questions": {
                        "route": {
                            "type": "choice",
                            "instructions": "x",
                            "criteria": {"a": "{}", "b": "not-json"},
                        }
                    },
                }
            },
            "valid JSON",
        ),
        ({"gold_unique_model_id": "not-a-candidate"}, "listed candidate"),
        (
            {"unique_model_id_probabilities": {"local/model-a": 1.0}},
            "every listed candidate ID",
        ),
    ],
)
def test_invalid_raw_contract_or_label_is_rejected(updates, message):
    source = record(**updates)
    if "unique_model_id_probabilities" in updates:
        source.pop("gold_unique_model_id")
    with pytest.raises(ValueError, match=message):
        normalize_autojev_route(source)


def test_rejects_ambiguous_or_malformed_soft_labels():
    both = record(
        unique_model_id_probabilities={
            "local/model-a": 0.1,
            "cloud/model-b": 0.7,
            "cloud/model-c": 0.2,
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        normalize_autojev_route(both)

    bad = record(
        unique_model_id_probabilities={
            "local/model-a": 0.1,
            "cloud/model-b": 0.7,
            "cloud/model-c": 0.3,
        }
    )
    bad.pop("gold_unique_model_id")
    with pytest.raises(ValueError, match="sum to one"):
        normalize_autojev_route(bad)


@pytest.mark.parametrize("group", ["", 0, None])
def test_rejects_malformed_explicit_group(group):
    with pytest.raises(ValueError, match="group must be a nonempty string"):
        normalize_autojev_route(record(group=group))
