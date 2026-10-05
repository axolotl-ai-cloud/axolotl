"""Hand-computed typed decision evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from axolotl.integrations.diffusion_decision.loss import (
    DistributionLabel,
    HardLabel,
    SetLabel,
)
from axolotl.integrations.diffusion_decision.metrics import (
    DecisionEvaluation,
    evaluate_decisions,
    paired_bootstrap,
)


def _row(
    record: str,
    question: str,
    probabilities: tuple[float, ...],
    target,
    *,
    source: str = "source",
    question_type: str = "choice",
    latency_ms: float = 10.0,
    semantic_labels: tuple[str, ...] | None = ("no", "yes", "unknown"),
) -> DecisionEvaluation:
    return DecisionEvaluation(
        record_id=record,
        question_id=question,
        source=source,
        question_type=question_type,
        allowed_ids=(10, 11, 12),
        probabilities=probabilities,
        target=target,
        latency_ms=latency_ms,
        semantic_labels=semantic_labels,
    )


def test_typed_metrics_match_hand_computations_and_keep_set_brier_separate():
    hard = _row("hard", "q", (0.7, 0.2, 0.1), HardLabel(0), latency_ms=10.0)
    dist = _row(
        "dist",
        "q",
        (0.2, 0.5, 0.3),
        DistributionLabel((0.1, 0.7, 0.2)),
        latency_ms=30.0,
        question_type="vote",
    )
    set_row = _row(
        "set",
        "q",
        (0.1, 0.3, 0.6),
        SetLabel((0, 1)),
        latency_ms=20.0,
        question_type="set",
    )

    report = evaluate_decisions((hard, dist, set_row))
    metrics = report["metrics"]

    expected_nll = (
        -math.log(0.7)
        - (0.1 * math.log(0.2) + 0.7 * math.log(0.5) + 0.2 * math.log(0.3))
        - math.log(0.4)
    ) / 3
    assert metrics["examples"] == 3
    assert metrics["questions"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["nll"] == pytest.approx(expected_nll)
    assert metrics["brier_hard_dist"] == pytest.approx((0.14 + 0.06) / 2)
    assert metrics["set_brier_uniform_support_proxy"] == pytest.approx(0.56)
    assert metrics["ece_15"] == pytest.approx((0.3 + 0.2 + 0.6) / 3)
    assert metrics["latency_ms_p50"] == pytest.approx(20.0)
    assert metrics["latency_ms_p95"] == pytest.approx(29.0)
    assert report["definitions"]["set_brier_uniform_support_proxy"].startswith(
        "squared error"
    )
    assert set(report["breakdown"]) == {"source:choice", "source:set", "source:vote"}


def test_metrics_average_questions_within_record_before_examples():
    many = tuple(
        _row("many", f"q{index}", (0.9, 0.05, 0.05), HardLabel(0))
        for index in range(20)
    )
    one = _row("one", "q", (0.9, 0.05, 0.05), HardLabel(1))

    metrics = evaluate_decisions((*many, one))["metrics"]

    assert metrics["accuracy"] == pytest.approx(0.5)


def test_macro_f1_requires_explicit_compatible_semantic_labels():
    first = _row("one", "q", (0.1, 0.8, 0.1), HardLabel(1))
    second = DecisionEvaluation(
        record_id="two",
        question_id="q",
        source="source",
        question_type="choice",
        allowed_ids=(20, 21, 22),
        probabilities=(0.8, 0.1, 0.1),
        target=HardLabel(0),
        latency_ms=10.0,
        semantic_labels=("yes", "no", "unknown"),
    )

    assert evaluate_decisions((first, second))["metrics"]["macro_f1"] == pytest.approx(
        1 / 3
    )
    assert (
        evaluate_decisions((replace(first, semantic_labels=None), second))["metrics"][
            "macro_f1"
        ]
        is None
    )
    incompatible = replace(second, semantic_labels=("true", "false", "unknown"))
    assert evaluate_decisions((first, incompatible))["metrics"]["macro_f1"] is None


def test_sparse_distribution_metrics_match_dense_target():
    row = _row(
        "sparse",
        "q",
        (0.2, 0.1, 0.7),
        DistributionLabel((0.7, 0.3), candidate_indices=(2, 0)),
    )
    dense = replace(row, target=DistributionLabel((0.3, 0.0, 0.7)))

    sparse_metrics = evaluate_decisions((row,))["metrics"]
    dense_metrics = evaluate_decisions((dense,))["metrics"]

    for name in ("accuracy", "nll", "brier_hard_dist", "ece_15", "macro_f1"):
        assert sparse_metrics[name] == pytest.approx(dense_metrics[name])


@pytest.mark.parametrize("indices", [(0, 0), (0, 3), (-1, 1)])
def test_sparse_distribution_rejects_invalid_candidate_indices(indices):
    row = _row(
        "invalid",
        "q",
        (0.2, 0.1, 0.7),
        DistributionLabel((0.7, 0.3), candidate_indices=indices),
    )

    with pytest.raises(ValueError, match="candidate indices|outside the allowed"):
        evaluate_decisions((row,))


def test_paired_bootstrap_clusters_questions_by_record():
    before = (
        _row("a", "q1", (0.8, 0.1, 0.1), HardLabel(0)),
        _row("a", "q2", (0.8, 0.1, 0.1), HardLabel(1)),
        _row("b", "q", (0.8, 0.1, 0.1), HardLabel(0)),
    )
    after = (
        before[0],
        replace(before[1], probabilities=(0.1, 0.8, 0.1)),
        before[2],
    )

    result = paired_bootstrap(before, after, seed=7, draws=200)

    assert result["metric"] == "accuracy"
    assert result["records"] == 2
    assert result["delta"] == pytest.approx(0.25)
    assert result["ci_95"][0] <= result["delta"] <= result["ci_95"][1]


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            (_row("one", "q", (float("nan"), 0.5, 0.5), HardLabel(0)),),
            "probabilities",
        ),
        (
            (_row("one", "q", (0.5, 0.25, 0.25), HardLabel(3)),),
            "gold_index",
        ),
        (
            (
                _row(
                    "one", "q", (0.5, 0.25, 0.25), HardLabel(0), latency_ms=float("inf")
                ),
            ),
            "latency_ms",
        ),
    ],
)
def test_metrics_reject_nonfinite_or_invalid_typed_rows(rows, message):
    with pytest.raises(ValueError, match=message):
        evaluate_decisions(rows)


def test_paired_bootstrap_rejects_misaligned_question_contracts():
    before = (_row("one", "q", (0.8, 0.1, 0.1), HardLabel(0)),)
    after = (replace(before[0], allowed_ids=(30, 31, 32)),)

    with pytest.raises(ValueError, match="question contracts"):
        paired_bootstrap(before, after, seed=0, draws=10)


def test_zero_probability_predictions_preserve_typed_nll_semantics():
    hard = _row("hard", "q", (1.0, 0.0, 0.0), HardLabel(0))
    dist = replace(hard, target=DistributionLabel((1.0, 0.0, 0.0)))
    set_row = replace(hard, target=SetLabel((0, 1)))
    for row in (hard, dist, set_row):
        assert evaluate_decisions((row,))["metrics"]["nll"] == 0
    impossible = replace(hard, target=HardLabel(1))
    assert math.isinf(evaluate_decisions((impossible,))["metrics"]["nll"])
    with pytest.raises(ValueError, match="finite metric deltas"):
        paired_bootstrap((impossible,), (impossible,), metric="nll", seed=0)


def test_logprobs_preserve_nll_when_probabilities_underflow():
    row = replace(
        _row("underflow", "q", (1.0, 0.0, 0.0), HardLabel(1)),
        restricted_logprobs=(0.0, -1000.0, -1001.0),
    )
    assert evaluate_decisions((row,))["metrics"]["nll"] == 1000.0
    set_row = replace(row, target=SetLabel((1, 2)))
    assert evaluate_decisions((set_row,))["metrics"]["nll"] == pytest.approx(
        1000.0 - math.log1p(math.exp(-1.0))
    )
    with pytest.raises(ValueError, match="match probabilities"):
        evaluate_decisions((replace(row, restricted_logprobs=(-1.0, -2.0, -3.0)),))


def test_unknown_bootstrap_metric_is_rejected():
    row = _row("a", "q", (0.8, 0.1, 0.1), HardLabel(0))
    with pytest.raises(ValueError, match="unknown paired bootstrap metric"):
        paired_bootstrap((row,), (row,), metric="typo", seed=0)


def test_independent_source_and_type_summaries_preserve_record_weighting():
    rows = (
        _row("shared", "q1", (0.8, 0.1, 0.1), HardLabel(0), source="alpha"),
        _row(
            "shared",
            "q2",
            (0.8, 0.1, 0.1),
            HardLabel(1),
            source="alpha",
            question_type="noul",
        ),
        _row("other", "q1", (0.8, 0.1, 0.1), HardLabel(0), source="alpha"),
        _row("shared", "q1", (0.8, 0.1, 0.1), HardLabel(1), source="beta"),
    )
    report = evaluate_decisions(rows)
    alpha = report["source_breakdown"]["alpha"]
    beta = report["source_breakdown"]["beta"]
    assert alpha["examples"] == 2
    assert alpha["questions"] == 3
    assert alpha["accuracy"] == pytest.approx(0.75)
    assert beta["examples"] == 1
    assert beta["accuracy"] == 0
    assert report["metrics"]["accuracy"] == pytest.approx(0.5)
    assert report["question_type_breakdown"]["choice"]["accuracy"] == pytest.approx(
        2 / 3
    )
    assert report["question_type_breakdown"]["noul"]["accuracy"] == 0
