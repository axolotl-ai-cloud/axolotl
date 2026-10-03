from __future__ import annotations

from copy import deepcopy

import pytest

from axolotl.integrations.decision import adapters
from axolotl.integrations.decision.adapters import (
    jsonl as jsonl_adapter,
    normalize_record,
)


def _row():
    return {
        "id": "row",
        "source": "local",
        "group": "group",
        "state": {"value": 1},
        "questions": {
            "q": {"type": "choice", "instructions": "pick", "options": ["a", "b"]}
        },
        "labels": {"q": {"kind": "hard", "gold_idx": 1}},
    }


def test_jsonl_dispatch_validates_once_and_does_not_mutate(monkeypatch):
    row = _row()
    original = deepcopy(row)
    calls = 0
    real = jsonl_adapter.normalize_jsonl

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setitem(adapters._ADAPTERS, "jsonl", counted)
    assert normalize_record("jsonl", row, training=False) == original
    assert row == original
    assert calls == 1


def test_jsonl_dispatch_preserves_malformed_error():
    row = _row()
    row["labels"] = {}
    with pytest.raises(ValueError, match="exactly one target"):
        normalize_record("jsonl", row, training=False)
