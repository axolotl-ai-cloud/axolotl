"""Normalized records reject targets incompatible with their question schema."""

from copy import deepcopy

import pytest

from axolotl.integrations.diffusion_decision.adapters.jsonl import normalize_jsonl


def record(target):
    return {
        "source": "local",
        "group": "family-1",
        "state": {"observation": 1},
        "questions": {"q": {"type": "choice", "options": ["act", "wait"]}},
        "labels": {"q": target},
    }


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "hard", "gold_idx": 1},
        {"kind": "dist", "probs": [0.2, 0.8]},
        {"kind": "set", "allowed_set": [0, 1]},
    ],
)
def test_all_label_kinds_roundtrip_without_mutating_input(target):
    source = record(target)
    original = deepcopy(source)
    result = normalize_jsonl(source)
    assert result == original
    result["questions"]["q"]["options"].append("other")
    assert source == original


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "hard", "gold_idx": -1},
        {"kind": "hard", "gold_idx": True},
        {"kind": "set", "allowed_set": []},
        {"kind": "set", "allowed_set": [0, 0]},
        {"kind": "dist", "probs": [0.8, 0.8]},
    ],
)
def test_invalid_targets_are_rejected(target):
    with pytest.raises(ValueError):
        normalize_jsonl(record(target))


def test_question_target_keys_must_match():
    source = record({"kind": "hard", "gold_idx": 0})
    source["labels"] = {}
    with pytest.raises(ValueError, match="exactly one"):
        normalize_jsonl(source)
