"""Example-balanced evaluation metrics for typed decision reads."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .loss import (
    DecisionLabelTarget,
    DistributionLabel,
    HardLabel,
    SetLabel,
)
from .records import OrdinalMetadata


@dataclass(frozen=True)
class DecisionEvaluation:
    """One question's restricted prediction and typed target."""

    record_id: str
    question_id: str
    source: str
    question_type: str
    allowed_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    target: DecisionLabelTarget
    latency_ms: float
    semantic_labels: tuple[str, ...] | None = None
    restricted_logprobs: tuple[float, ...] | None = None
    ordinal_metadata: OrdinalMetadata | None = None


MetricName = Literal[
    "accuracy",
    "nll",
    "brier_hard_dist",
    "set_brier_uniform_support_proxy",
    "rps",
]


@dataclass(frozen=True)
class _QuestionMetrics:
    accuracy: float
    nll: float
    brier_hard_dist: float | None
    set_brier_uniform_support_proxy: float | None
    rps: float | None
    confidence: float
    ece_target: float


@dataclass(frozen=True)
class _RecordMetrics:
    accuracy: float
    nll: float
    brier_hard_dist: float | None
    set_brier_uniform_support_proxy: float | None
    rps: float | None


def evaluate_decisions(
    rows: Sequence[DecisionEvaluation], *, ece_bins: int = 15
) -> dict[str, Any]:
    """Summarize typed reads, averaging questions within each record first.

    ECE compares top-label confidence with the target probability assigned to the
    predicted label. For hard and set labels this is binary correctness. Set Brier
    is reported separately as a uniform-support proxy rather than calibration.
    """
    _validate_rows(rows)
    _validate_bins(ece_bins)
    grouped: dict[str, list[DecisionEvaluation]] = defaultdict(list)
    sources: dict[str, list[DecisionEvaluation]] = defaultdict(list)
    question_types: dict[str, list[DecisionEvaluation]] = defaultdict(list)
    for row in rows:
        grouped[f"{row.source}:{row.question_type}"].append(row)
        sources[row.source].append(row)
        question_types[row.question_type].append(row)
    return {
        "metrics": _summary(rows, ece_bins),
        "breakdown": {
            name: _summary(source_rows, ece_bins)
            for name, source_rows in sorted(grouped.items())
        },
        "source_breakdown": {
            name: _summary(source_rows, ece_bins)
            for name, source_rows in sorted(sources.items())
        },
        "question_type_breakdown": {
            name: _summary(type_rows, ece_bins)
            for name, type_rows in sorted(question_types.items())
        },
        "definitions": {
            "aggregation": "mean questions within each record, then mean records",
            "dist_accuracy": "prediction equals the target distribution majority label",
            "ece": f"{ece_bins}-bin top-label confidence versus target probability at the predicted label",
            "set_brier_uniform_support_proxy": "squared error to a uniform distribution over correct set members; not calibration",
            "macro_f1": "reported only for compatible explicit semantic label identities",
            "rps": "score-only normalized ranked probability score using source-ordered criteria",
        },
    }


def paired_bootstrap(
    before: Sequence[DecisionEvaluation],
    after: Sequence[DecisionEvaluation],
    *,
    metric: MetricName = "accuracy",
    seed: int,
    draws: int = 10_000,
) -> dict[str, Any]:
    """Return a record-clustered bootstrap interval for a paired metric delta."""
    if metric not in {
        "accuracy",
        "nll",
        "brier_hard_dist",
        "set_brier_uniform_support_proxy",
        "rps",
    }:
        raise ValueError("unknown paired bootstrap metric")
    _validate_rows(before)
    _validate_rows(after)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    before_by_question = _rows_by_question(before)
    after_by_question = _rows_by_question(after)
    if before_by_question.keys() != after_by_question.keys():
        raise ValueError("paired bootstrap requires identical record and question IDs")
    for key, left in before_by_question.items():
        right = after_by_question[key]
        if not _same_question_contract(left, right):
            raise ValueError("paired bootstrap requires identical question contracts")

    before_records = _metric_by_record(before, metric)
    after_records = _metric_by_record(after, metric)
    if before_records.keys() != after_records.keys() or not before_records:
        raise ValueError("paired bootstrap has no compatible records for the metric")
    keys = tuple(sorted(before_records))
    deltas = tuple(after_records[key] - before_records[key] for key in keys)
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("paired bootstrap requires finite metric deltas")
    generator = random.Random(seed)  # nosec B311 - fixed reporting bootstrap.
    samples = sorted(
        sum(deltas[generator.randrange(len(deltas))] for _ in keys) / len(keys)
        for _ in range(draws)
    )
    return {
        "metric": metric,
        "records": len(keys),
        "draws": draws,
        "delta": sum(deltas) / len(deltas),
        "ci_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
    }


def _summary(rows: Sequence[DecisionEvaluation], ece_bins: int) -> dict[str, Any]:
    grouped = _rows_by_record(rows)
    record_metrics = [_record_metrics(value) for value in grouped.values()]
    question_metrics = [_question_metrics(row) for row in rows]
    ece = _ece(grouped.values(), ece_bins)
    latency = [_record_latency(value) for value in grouped.values()]
    return {
        "examples": len(grouped),
        "questions": len(rows),
        "accuracy": _mean(metrics.accuracy for metrics in record_metrics),
        "nll": _mean(metrics.nll for metrics in record_metrics),
        "brier_hard_dist": _mean_optional(
            metrics.brier_hard_dist for metrics in record_metrics
        ),
        "set_brier_uniform_support_proxy": _mean_optional(
            metrics.set_brier_uniform_support_proxy for metrics in record_metrics
        ),
        "rps": _mean_optional(metrics.rps for metrics in record_metrics),
        "rps_eligible_questions": sum(
            metric.rps is not None for metric in question_metrics
        ),
        "rps_eligible_records": sum(
            metric.rps is not None for metric in record_metrics
        ),
        "ece_15" if ece_bins == 15 else f"ece_{ece_bins}": ece,
        "latency_ms_p50": _percentile(latency, 0.5),
        "latency_ms_p95": _percentile(latency, 0.95),
        "macro_f1": _macro_f1(grouped.values()),
    }


def _record_metrics(rows: Sequence[DecisionEvaluation]) -> _RecordMetrics:
    question_metrics = [_question_metrics(row) for row in rows]
    return _RecordMetrics(
        accuracy=_mean(metric.accuracy for metric in question_metrics),
        nll=_mean(metric.nll for metric in question_metrics),
        brier_hard_dist=_mean_optional(
            metric.brier_hard_dist for metric in question_metrics
        ),
        set_brier_uniform_support_proxy=_mean_optional(
            metric.set_brier_uniform_support_proxy for metric in question_metrics
        ),
        rps=_mean_optional(metric.rps for metric in question_metrics),
    )


def _metric_by_record(
    rows: Sequence[DecisionEvaluation], metric: MetricName
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for key, record_rows in _rows_by_record(rows).items():
        value = _record_metric_value(_record_metrics(record_rows), metric)
        if value is not None:
            values[key] = value
    return values


def _record_metric_value(metrics: _RecordMetrics, metric: MetricName) -> float | None:
    if metric == "accuracy":
        return metrics.accuracy
    if metric == "nll":
        return metrics.nll
    if metric == "brier_hard_dist":
        return metrics.brier_hard_dist
    if metric == "set_brier_uniform_support_proxy":
        return metrics.set_brier_uniform_support_proxy
    return metrics.rps


def _question_metrics(row: DecisionEvaluation) -> _QuestionMetrics:
    prediction = _argmax(row.probabilities)
    confidence = row.probabilities[prediction]
    logprobs = row.restricted_logprobs or tuple(
        math.log(value) if value else -math.inf for value in row.probabilities
    )
    if isinstance(row.target, HardLabel):
        target = _one_hot(len(row.probabilities), row.target.gold_index)
        return _QuestionMetrics(
            accuracy=float(prediction == row.target.gold_index),
            nll=-logprobs[row.target.gold_index],
            brier_hard_dist=_squared_error(row.probabilities, target),
            set_brier_uniform_support_proxy=None,
            rps=_rps(row, target),
            confidence=confidence,
            ece_target=float(prediction == row.target.gold_index),
        )
    if isinstance(row.target, DistributionLabel):
        target = _distribution_target(row.target, len(row.probabilities))
        majority = _argmax(target)
        return _QuestionMetrics(
            accuracy=float(prediction == majority),
            nll=-sum(
                value * logprob
                for value, logprob in zip(target, logprobs, strict=True)
                if value > 0
            ),
            brier_hard_dist=_squared_error(row.probabilities, target),
            set_brier_uniform_support_proxy=None,
            rps=_rps(row, target),
            confidence=confidence,
            ece_target=target[prediction],
        )
    assert isinstance(row.target, SetLabel)
    correct = set(row.target.allowed_indices)
    selected_logprobs = tuple(logprobs[index] for index in correct)
    maximum = max(selected_logprobs)
    logmass = (
        maximum
        + math.log(sum(math.exp(value - maximum) for value in selected_logprobs))
        if math.isfinite(maximum)
        else -math.inf
    )
    uniform = tuple(
        1 / len(correct) if index in correct else 0.0
        for index in range(len(row.probabilities))
    )
    return _QuestionMetrics(
        accuracy=float(prediction in correct),
        nll=-logmass,
        brier_hard_dist=None,
        set_brier_uniform_support_proxy=_squared_error(row.probabilities, uniform),
        rps=None,
        confidence=confidence,
        ece_target=float(prediction in correct),
    )


def _distribution_target(target: DistributionLabel, count: int) -> tuple[float, ...]:
    if target.other_probability:
        raise ValueError(
            "restricted evaluation does not support dist other_probability"
        )
    if target.candidate_indices is None:
        return target.probabilities
    result = [0.0] * count
    for index, probability in zip(
        target.candidate_indices, target.probabilities, strict=True
    ):
        result[index] = probability
    return tuple(result)


def _ece(records: Iterable[Sequence[DecisionEvaluation]], bins: int) -> float:
    bucket_weight = [0.0] * bins
    bucket_confidence = [0.0] * bins
    bucket_target = [0.0] * bins
    total_weight = 0.0
    for rows in records:
        weight = 1 / len(rows)
        for row in rows:
            metrics = _question_metrics(row)
            confidence = metrics.confidence
            bucket = min(int(confidence * bins), bins - 1)
            bucket_weight[bucket] += weight
            bucket_confidence[bucket] += weight * confidence
            bucket_target[bucket] += weight * metrics.ece_target
            total_weight += weight
    return sum(
        weight
        / total_weight
        * abs(bucket_confidence[index] / weight - bucket_target[index] / weight)
        for index, weight in enumerate(bucket_weight)
        if weight
    )


def _macro_f1(records: Iterable[Sequence[DecisionEvaluation]]) -> float | None:
    rows = [row for record_rows in records for row in record_rows]
    eligible = [
        row for row in rows if isinstance(row.target, (HardLabel, DistributionLabel))
    ]
    if not eligible or any(row.semantic_labels is None for row in eligible):
        return None
    first_labels = eligible[0].semantic_labels
    if first_labels is None:
        return None
    label_space = frozenset(first_labels)
    if not label_space or any(
        row.semantic_labels is None or frozenset(row.semantic_labels) != label_space
        for row in eligible
    ):
        return None
    weighted_counts: dict[str, dict[str, float]] = {
        label: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for label in label_space
    }
    by_record = _rows_by_record(eligible)
    for record_rows in by_record.values():
        weight = 1 / len(record_rows)
        for row in record_rows:
            assert row.semantic_labels is not None
            prediction = row.semantic_labels[_argmax(row.probabilities)]
            if isinstance(row.target, HardLabel):
                target_index = row.target.gold_index
            else:
                assert isinstance(row.target, DistributionLabel)
                target_index = _argmax(
                    _distribution_target(row.target, len(row.probabilities))
                )
            target = row.semantic_labels[target_index]
            for label in label_space:
                if prediction == label and target == label:
                    weighted_counts[label]["tp"] += weight
                elif prediction == label:
                    weighted_counts[label]["fp"] += weight
                elif target == label:
                    weighted_counts[label]["fn"] += weight
    values = []
    for counts in weighted_counts.values():
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        values.append(0.0 if not denominator else 2 * counts["tp"] / denominator)
    return _mean(values)


def _rows_by_record(
    rows: Sequence[DecisionEvaluation],
) -> dict[tuple[str, str], list[DecisionEvaluation]]:
    grouped: dict[tuple[str, str], list[DecisionEvaluation]] = defaultdict(list)
    for row in rows:
        grouped[(row.source, row.record_id)].append(row)
    return grouped


def _rows_by_question(
    rows: Sequence[DecisionEvaluation],
) -> dict[tuple[str, str, str], DecisionEvaluation]:
    result: dict[tuple[str, str, str], DecisionEvaluation] = {}
    for row in rows:
        key = (row.source, row.record_id, row.question_id)
        if key in result:
            raise ValueError("paired bootstrap requires unique questions per record")
        result[key] = row
    return result


def _same_question_contract(
    left: DecisionEvaluation, right: DecisionEvaluation
) -> bool:
    return (
        left.question_type == right.question_type
        and left.allowed_ids == right.allowed_ids
        and left.target == right.target
        and left.semantic_labels == right.semantic_labels
        and left.ordinal_metadata == right.ordinal_metadata
    )


def _record_latency(rows: Sequence[DecisionEvaluation]) -> float:
    latency = rows[0].latency_ms
    if any(
        not math.isclose(row.latency_ms, latency, rel_tol=0.0, abs_tol=1e-9)
        for row in rows[1:]
    ):
        raise ValueError("questions in one record must share one decision latency")
    return latency


def _validate_rows(rows: Sequence[DecisionEvaluation]) -> None:
    if not rows:
        raise ValueError("cannot evaluate an empty decision split")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        _validate_row(row)
        key = (row.source, row.record_id, row.question_id)
        if key in seen:
            raise ValueError("decision metrics require unique questions per record")
        seen.add(key)


def _validate_row(row: DecisionEvaluation) -> None:
    for name, value in (
        ("record_id", row.record_id),
        ("question_id", row.question_id),
        ("source", row.source),
        ("question_type", row.question_type),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    count = len(row.allowed_ids)
    if not count or len(row.probabilities) != count:
        raise ValueError("allowed IDs and probabilities must align and be nonempty")
    if len(set(row.allowed_ids)) != count or any(
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier < 0
        for identifier in row.allowed_ids
    ):
        raise ValueError("allowed IDs must be unique nonnegative integers")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
        for value in row.probabilities
    ) or not math.isclose(sum(row.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities must be finite, nonnegative, and sum to one")
    if row.restricted_logprobs is not None:
        if len(row.restricted_logprobs) != count or any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or math.isnan(value)
            or value > 0
            for value in row.restricted_logprobs
        ):
            raise ValueError("restricted logprobs must align and be nonpositive")
        if any(
            not math.isclose(math.exp(logprob), probability, rel_tol=1e-5, abs_tol=1e-7)
            for logprob, probability in zip(
                row.restricted_logprobs, row.probabilities, strict=True
            )
        ):
            raise ValueError("restricted logprobs must match probabilities")
    if (
        isinstance(row.latency_ms, bool)
        or not isinstance(row.latency_ms, (float, int))
        or not math.isfinite(row.latency_ms)
        or row.latency_ms < 0
    ):
        raise ValueError("latency_ms must be finite and nonnegative")
    _validate_target(row.target, count)
    if row.semantic_labels is not None and (
        len(row.semantic_labels) != count
        or len(set(row.semantic_labels)) != count
        or any(not isinstance(value, str) or not value for value in row.semantic_labels)
    ):
        raise ValueError(
            "semantic labels must be distinct nonempty strings per candidate"
        )
    if row.ordinal_metadata is not None:
        _validate_ordinal_metadata(row.ordinal_metadata, count)
        if row.question_type != "score":
            raise ValueError("ordinal metadata is only valid for score questions")


def _validate_ordinal_metadata(value: OrdinalMetadata, count: int) -> None:
    if count < 2:
        raise ValueError("ordinal metadata requires at least two candidates")
    if not (
        len(value.levels)
        == len(value.source_ids)
        == len(value.candidate_ranks)
        == count
    ):
        raise ValueError("ordinal metadata must align with allowed candidates")
    if (
        any(not isinstance(item, str) or not item for item in value.levels)
        or any(not isinstance(item, str) or not item for item in value.source_ids)
        or len(set(value.source_ids)) != count
    ):
        raise ValueError(
            "ordinal levels and source IDs must be nonempty and source IDs unique"
        )
    ranks = tuple(value.candidate_ranks)
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
        raise ValueError("ordinal candidate ranks must be integers")
    if set(ranks) != set(range(count)):
        raise ValueError("ordinal candidate ranks must be a zero-based bijection")


def _rps(row: DecisionEvaluation, target: Sequence[float]) -> float | None:
    metadata = row.ordinal_metadata
    if metadata is None or row.question_type != "score":
        return None
    predicted = _rank_ordered(row.probabilities, metadata.candidate_ranks)
    expected = _rank_ordered(target, metadata.candidate_ranks)
    predicted_cdf = 0.0
    expected_cdf = 0.0
    squared = 0.0
    for probability, target_probability in zip(
        predicted[:-1], expected[:-1], strict=True
    ):
        predicted_cdf += probability
        expected_cdf += target_probability
        squared += (predicted_cdf - expected_cdf) ** 2
    return squared / (len(predicted) - 1)


def _rank_ordered(values: Sequence[float], ranks: Sequence[int]) -> tuple[float, ...]:
    ordered = [0.0] * len(values)
    for value, rank in zip(values, ranks, strict=True):
        ordered[rank] = value
    return tuple(ordered)


def _validate_target(target: DecisionLabelTarget, count: int) -> None:
    if isinstance(target, HardLabel):
        _validate_index(target.gold_index, count, "gold_index")
        return
    if isinstance(target, DistributionLabel):
        indices = target.candidate_indices
        if (
            not target.probabilities
            or len(target.probabilities) != (count if indices is None else len(indices))
            or any(
                isinstance(value, bool) or not math.isfinite(value) or value < 0
                for value in target.probabilities
            )
            or not math.isclose(
                sum(target.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise ValueError(
                "distribution targets must be finite and align with candidates"
            )
        if indices is not None:
            if len(set(indices)) != len(indices):
                raise ValueError(
                    "distribution targets require unique candidate indices"
                )
            for index in indices:
                _validate_index(index, count, "distribution target index")
        return
    if isinstance(target, SetLabel):
        if not target.allowed_indices or len(set(target.allowed_indices)) != len(
            target.allowed_indices
        ):
            raise ValueError("set targets require unique allowed indices")
        for index in target.allowed_indices:
            _validate_index(index, count, "set target index")
        return
    raise TypeError("decision evaluation requires a typed label target")


def _validate_index(index: int, count: int, name: str) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError(f"{name} is outside the allowed candidates")


def _validate_bins(bins: int) -> None:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("ECE bins must be a positive integer")


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _one_hot(count: int, index: int) -> tuple[float, ...]:
    return tuple(float(candidate == index) for candidate in range(count))


def _squared_error(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((first - second) ** 2 for first, second in zip(left, right, strict=True))


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("metric requires at least one value")
    return sum(materialized) / len(materialized)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    materialized = tuple(value for value in values if value is not None)
    return None if not materialized else _mean(materialized)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower, upper = math.floor(rank), math.ceil(rank)
    return (
        ordered[lower]
        if lower == upper
        else ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
    )
