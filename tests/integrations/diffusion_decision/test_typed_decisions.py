"""Native multi-question schemas preserve option semantics and soft targets."""

import json

import pytest
import torch

from axolotl.integrations.diffusion_decision.adapters import normalize_record
from axolotl.integrations.diffusion_decision.adapters.typed_decisions import normalize
from axolotl.integrations.diffusion_decision.loss import (
    DecisionLabelExample,
    DecisionLabelQuestion,
    decision_label_loss,
    label_target_from_mapping,
)


def sample():
    return {
        "id": "a",
        "state": '{"trace":"observed"}',
        "workflow": "trace_review",
        "questions": json.dumps(
            {
                "s": {"type": "score", "criteria": ["low", "high"]},
                "c": {
                    "type": "choice",
                    "criteria": {"act": "Proceed", "wait": "Delay"},
                },
                "b": {
                    "type": "noul",
                    "criteria": {"true": "Supported", "false": "Refuted"},
                },
            }
        ),
        "gold": json.dumps(
            {
                "s": {"probabilities": {"1": 0.8, "0": 0.2}},
                "c": {"probabilities": {"wait": 0.4, "act": 0.6}},
                "b": {"probabilities": {"false": 1, "true": 0}},
            }
        ),
    }


def test_native_multi_question_preserves_all_semantics():
    result = normalize(sample())
    assert result["questions"]["s"]["levels"] == ["low", "high"]
    assert result["labels"]["s"] == {
        "kind": "dist",
        "candidate_indices": [0, 1],
        "candidate_ids": ["0", "1"],
        "probs": [0.2, 0.8],
        "other_probability": 0.0,
    }
    assert result["questions"]["c"]["options"] == [
        {"name": "act", "description": "Proceed"},
        {"name": "wait", "description": "Delay"},
    ]
    assert result["labels"]["c"]["probs"] == [0.6, 0.4]
    assert result["labels"]["b"] == {
        "kind": "dist",
        "candidate_indices": [1],
        "candidate_ids": ["no"],
        "probs": [1.0],
        "other_probability": 0.0,
    }
    assert result["questions"]["b"]["criteria"]["true"] == "Supported"
    assert result["family"] == "trace_review"
    assert result["state"] == {"trace": "observed"}
    assert result["source_metadata"]["provenance"] == {
        "dataset": "LocalLLaMA/typed-decisions",
        "workflow": "trace_review",
        "split": None,
        "row_split": None,
        "source_id": "a",
    }


def test_missing_probability_rejected_without_inventing_mass():
    row = sample()
    gold = json.loads(row["gold"])
    del gold["s"]["probabilities"]["0"]
    row["gold"] = json.dumps(gold)
    with pytest.raises(ValueError, match="each alternative"):
        normalize(row)


def test_more_than_four_soft_alternatives_preserve_full_distribution():
    row = sample()
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    questions["s"]["criteria"] = ["a", "b", "c", "d", "e"]
    gold["s"]["probabilities"] = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.15, "4": 0.25}
    row["questions"] = json.dumps(questions)
    row["gold"] = json.dumps(gold)
    target = normalize(row)["labels"]["s"]
    assert target == {
        "kind": "dist",
        "candidate_indices": [0, 1, 2, 3, 4],
        "candidate_ids": ["0", "1", "2", "3", "4"],
        "probs": [0.1, 0.2, 0.3, 0.15, 0.25],
        "other_probability": 0.0,
    }
    result = decision_label_loss(
        torch.zeros((1, 1, 5)),
        [
            DecisionLabelExample(
                (
                    DecisionLabelQuestion(
                        0,
                        (0, 1, 2, 3, 4),
                        label_target_from_mapping(target),
                    ),
                )
            )
        ],
        torch.tensor([[True]]),
        label_softmax="both",
    )
    assert torch.isfinite(result.loss)


def test_missing_question_target_rejected():
    row = sample()
    row["gold"] = "{}"
    with pytest.raises(ValueError, match="exactly one target"):
        normalize(row)


def test_training_is_fail_closed_to_declared_train_split():
    result = normalize_record(
        "diffusion_decision.typed_decisions",
        sample(),
        training=True,
        source_split="train",
    )
    assert result["source_metadata"]["provenance"]["split"] == "train"
    with pytest.raises(ValueError, match="declared train split"):
        normalize_record(
            "diffusion_decision.typed_decisions",
            sample(),
            training=True,
            source_split="test",
        )


def test_row_split_mismatch_is_rejected():
    row = sample()
    row["split"] = "test"
    with pytest.raises(ValueError, match="disagrees"):
        normalize_record(
            "diffusion_decision.typed_decisions",
            row,
            training=True,
            source_split="train",
        )


def test_native_hard_smoothing_preserves_question_semantics():
    row = sample()
    gold = json.loads(row["gold"])
    gold["b"]["smoothing"] = 0.1
    row["gold"] = json.dumps(gold)
    baseline = normalize(sample())
    result = normalize(row)
    assert result["questions"] == baseline["questions"]
    assert result["labels"]["s"] == baseline["labels"]["s"]
    assert result["labels"]["c"] == baseline["labels"]["c"]
    assert result["labels"]["b"] == {"kind": "hard", "gold_idx": 1, "smoothing": 0.1}


@pytest.mark.parametrize("smoothing", [True, -0.1, 1.0, float("nan")])
def test_native_hard_smoothing_validates_metadata(smoothing):
    row = sample()
    gold = json.loads(row["gold"])
    gold["b"]["smoothing"] = smoothing
    row["gold"] = json.dumps(gold)
    with pytest.raises(ValueError, match="hard smoothing"):
        normalize(row)


def test_native_hard_smoothing_rejects_soft_targets():
    row = sample()
    gold = json.loads(row["gold"])
    gold["s"]["smoothing"] = 0.1
    row["gold"] = json.dumps(gold)
    with pytest.raises(ValueError, match="one-hot"):
        normalize(row)


def test_native_hard_smoothing_uses_one_hot_probability_not_gold_label():
    row = sample()
    gold = json.loads(row["gold"])
    gold["b"]["label"] = "true"
    gold["b"]["smoothing"] = 0.1
    row["gold"] = json.dumps(gold)
    assert normalize(row)["labels"]["b"] == {
        "kind": "hard",
        "gold_idx": 1,
        "smoothing": 0.1,
    }
