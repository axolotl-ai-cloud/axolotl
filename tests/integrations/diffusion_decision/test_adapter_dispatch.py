"""Public source dispatch enforces shared validation and evaluation-only data."""

import pytest

from axolotl.integrations.diffusion_decision.adapters import normalize_record

from tests.integrations.diffusion_decision.helpers import (
    make_record,
)


def local_record(source="local"):
    return make_record(
        None, source=source, group="a", state="observed", question_type="noul"
    )


def test_prefixed_and_short_names_are_equivalent():
    row = local_record()
    assert normalize_record(
        "diffusion_decision.jsonl", row, training=True
    ) == normalize_record("jsonl", row, training=True)


def test_eval_corpus_cannot_enter_training_through_jsonl():
    with pytest.raises(ValueError, match="reserved for evaluation"):
        normalize_record("jsonl", local_record("typed_decisions"), training=True)
    assert (
        normalize_record("jsonl", local_record("typed_decisions"), training=False)[
            "source"
        ]
        == "typed_decisions"
    )


def test_eval_corpus_adapter_rejected_before_reading_row():
    with pytest.raises(ValueError, match="declared train split"):
        normalize_record("typed_decisions", {}, training=True)


def test_explicit_typed_train_adapter_is_allowed():
    row = {
        "id": "a",
        "workflow": "workflow",
        "state": "{}",
        "questions": '{"q":{"type":"noul"}}',
        "gold": '{"q":{"probabilities":{"false":0.0,"true":1.0}}}',
    }
    assert (
        normalize_record("typed_decisions", row, training=True, source_split="train")[
            "source"
        ]
        == "typed_decisions"
    )


def test_unknown_adapter_rejected():
    with pytest.raises(ValueError, match="unknown decision adapter"):
        normalize_record("typo", {}, training=False)
