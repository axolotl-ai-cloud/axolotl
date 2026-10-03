"""Teacher uncertainty is retained, including deterministic teacher outputs."""

import pytest

from axolotl.integrations.decision.adapters.jev_distill import (
    normalize_jev_distill,
)


def source_row(**changes):
    row = {
        "id": "teacher-1",
        "kind": "score",
        "options": ["0", "1", "2"],
        "target": [0.1, 0.7, 0.2],
        "state": "Observed evidence",
        "question": "Rate reliability",
        "domain": "review",
        "family": "knowledge",
        "source": "teacher-version",
    }
    return row | changes


def test_score_keeps_teacher_distribution_and_semantic_levels():
    result = normalize_jev_distill(source_row())
    assert result["questions"]["q1"]["levels"] == ["0", "1", "2"]
    assert result["labels"]["q1"] == {"kind": "dist", "probs": [0.1, 0.7, 0.2]}
    assert result["family"] == "knowledge"
    assert result["source_metadata"]["source"] == "teacher-version"
    assert result["state"] == "Observed evidence"


def test_boolean_identity_remapping_and_one_hot_teacher_kind():
    result = normalize_jev_distill(
        source_row(kind="noul", options=["false", "true"], target=[1, 0])
    )
    assert result["labels"]["q1"] == {"kind": "dist", "probs": [0.0, 1.0]}


@pytest.mark.parametrize("target", [[0.2, 0.3], [0.1, 0.2, 0.3], [float("nan"), 0, 1]])
def test_invalid_teacher_probability_vectors_fail(target):
    with pytest.raises(ValueError):
        normalize_jev_distill(source_row(target=target))
