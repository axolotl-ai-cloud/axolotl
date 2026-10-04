"""Prepared-canvas evaluation and artifact serialization for decision readers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import torch

from .loss import (
    DecisionLabelTarget,
    DistributionLabel,
    HardLabel,
    SetLabel,
    label_target_from_mapping,
)
from .metrics import DecisionEvaluation, evaluate_decisions
from .preprocessing import ordinal_metadata_for_canvas
from .readers.base import DecisionRead
from .records import DecisionCanvas, OrdinalMetadata

LOG = logging.getLogger(__name__)


class PreparedDecisionReader(Protocol):
    """Reader shape used by the generic prepared-canvas evaluator."""

    def read(
        self,
        model: Any,
        spec: Any,
        canvas: DecisionCanvas,
        *,
        steps: int,
        seed: int,
        hold_label_noise: bool = False,
        diagnostics: bool,
    ) -> DecisionRead: ...


@dataclass(frozen=True)
class PreparedEvaluationRun:
    """Compact predictions and typed rows emitted from immutable prepared canvases."""

    rows: tuple[DecisionEvaluation, ...]
    predictions: tuple[dict[str, Any], ...]
    ordinal_metadata: bool = False
    read_stats: Mapping[str, Any] | None = None


def prepared_rows_canvas_sha256(
    rows: Sequence[Mapping[str, Any]], *, ordinal_metadata: bool = False
) -> str:
    """Hash every immutable canvas and supervision field before a prepared read."""
    value = []
    for row in rows:
        prepared = _prepared_row(row, ordinal_metadata=ordinal_metadata)
        canvas = prepared["canvas"]
        assert isinstance(canvas, DecisionCanvas)
        value.append(
            {
                "record_id": prepared["record_id"],
                "source": prepared["source"],
                "question_types": prepared["question_types"],
                "targets": [_target_mapping(target) for target in prepared["targets"]],
                "canvas": _canvas_mapping(canvas, ordinal_metadata=ordinal_metadata),
            }
        )
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canvas_mapping(
    canvas: DecisionCanvas, *, ordinal_metadata: bool = False
) -> dict[str, Any]:
    value = {
        "prompt_ids": list(canvas.prompt_ids),
        "canvas_ids": list(canvas.canvas_ids),
        "label_positions": list(canvas.label_positions),
        "allowed_ids": [list(value) for value in canvas.allowed_ids],
        "question_ids": list(canvas.question_ids),
        "targets": [_target_mapping(_target(value)) for value in canvas.targets],
        "pinned_mask": list(canvas.pinned_mask),
        "semantic_mask": list(canvas.semantic_mask),
        "slot_mask": list(canvas.slot_mask),
        "template_length": canvas.template_length,
        "prompt_slot_mask": list(canvas.prompt_slot_mask),
    }
    if ordinal_metadata:
        value["ordinal_metadata"] = _ordinal_metadata_mapping(canvas.ordinal_metadata)
    return value


def _prediction_canvas_mapping(
    canvas: DecisionCanvas, *, ordinal_metadata: bool = False
) -> dict[str, Any]:
    value = {
        "prompt_ids": list(canvas.prompt_ids),
        "canvas_ids": list(canvas.canvas_ids),
        "label_positions": list(canvas.label_positions),
        "allowed_ids": [list(value) for value in canvas.allowed_ids],
        "pinned_mask": list(canvas.pinned_mask),
        "semantic_mask": list(canvas.semantic_mask),
        "template_length": canvas.template_length,
    }
    if ordinal_metadata:
        value["ordinal_metadata"] = _ordinal_metadata_mapping(canvas.ordinal_metadata)
    return value


def evaluate_prepared_rows(
    reader: PreparedDecisionReader,
    model: Any,
    spec: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int = 1,
    seed: int = 0,
    warmup: int = 0,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    ordinal_metadata: bool = False,
    codebook: str = "vendored26",
    hold_label_noise: bool = False,
    batch_size: int = 1,
    max_batch_tokens: int | None = None,
) -> PreparedEvaluationRun:
    """Read exact prepared canvases without rebuilding templates, noise, or targets."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("prepared evaluation steps must be a positive integer")
    if not isinstance(hold_label_noise, bool):
        raise TypeError("hold_label_noise must be bool")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("evaluation seed must be an integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if max_batch_tokens is not None and (
        isinstance(max_batch_tokens, bool)
        or not isinstance(max_batch_tokens, int)
        or max_batch_tokens < 1
    ):
        raise ValueError("max_batch_tokens must be a positive integer or None")
    if not rows:
        raise ValueError("prepared evaluation requires at least one row")
    sync = synchronize or (lambda: None)
    prepared = tuple(
        _prepared_row(row, ordinal_metadata=ordinal_metadata, codebook=codebook)
        for row in rows
    )
    _validate_prepared_questions(prepared)

    def read(canvas: DecisionCanvas, index: int) -> DecisionRead:
        kwargs: dict[str, Any] = {
            "steps": steps,
            "seed": seed + index,
            "diagnostics": True,
        }
        if hold_label_noise:
            kwargs["hold_label_noise"] = True
        return reader.read(model, spec, canvas, **kwargs)

    def read_many(
        start: int, values: Sequence[Mapping[str, Any]]
    ) -> tuple[DecisionRead, ...]:
        if len(values) == 1:
            return (read(values[0]["canvas"], start),)
        batch_reader = getattr(reader, "read_batch", None)
        if not callable(batch_reader):
            raise NotImplementedError("reader does not support batched decision reads")
        kwargs: dict[str, Any] = {
            "steps": steps,
            "seeds": tuple(seed + start + offset for offset in range(len(values))),
            "diagnostics": True,
        }
        if hold_label_noise:
            kwargs["hold_label_noise"] = True
        result = tuple(
            batch_reader(
                model, spec, tuple(value["canvas"] for value in values), **kwargs
            )
        )
        if len(result) != len(values):
            raise ValueError("batched reader returned the wrong number of reads")
        return result

    def batches(limit: int):
        start = 0
        while start < limit:
            stop = start
            tokens = 0
            while stop < limit and stop - start < batch_size:
                canvas = prepared[stop]["canvas"]
                assert isinstance(canvas, DecisionCanvas)
                width = len(canvas.prompt_ids) + len(canvas.canvas_ids)
                if (
                    stop > start
                    and max_batch_tokens is not None
                    and tokens + width > max_batch_tokens
                ):
                    break
                tokens += width
                stop += 1
            yield start, prepared[start:stop]
            start = stop

    warmup_count = min(warmup, len(prepared))
    for start, batch in batches(warmup_count):
        read_many(start, batch)
    evaluations: list[DecisionEvaluation] = []
    predictions: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []
    batch_tokens: list[int] = []
    previous_completed = 0
    for start, batch in batches(len(prepared)):
        sync()
        started = clock()
        results = read_many(start, batch)
        sync()
        latency_ms = (clock() - started) * 1_000
        batch_latencies_ms.append(latency_ms)
        batch_tokens.append(
            sum(
                len(value["canvas"].prompt_ids) + len(value["canvas"].canvas_ids)
                for value in batch
            )
        )
        for offset, (prepared_row, result) in enumerate(
            zip(batch, results, strict=True)
        ):
            produced = _read_rows(
                prepared_row,
                result,
                seed + start + offset,
                steps,
                latency_ms,
                ordinal_metadata=ordinal_metadata,
            )
            evaluations.extend(item[0] for item in produced)
            predictions.extend(item[1] for item in produced)
        completed = start + len(batch)
        if completed // 100 > previous_completed // 100 or completed == len(prepared):
            LOG.info("Evaluated decision reads: %s/%s", completed, len(prepared))
        previous_completed = completed
    read_seconds = sum(batch_latencies_ms) / 1_000
    tokens = sum(batch_tokens)
    return PreparedEvaluationRun(
        tuple(evaluations),
        tuple(predictions),
        ordinal_metadata=ordinal_metadata,
        read_stats={
            "records": len(prepared),
            "batches": len(batch_tokens),
            "tokens": tokens,
            "max_batch_tokens": max(batch_tokens),
            "mean_batch_tokens": tokens / len(batch_tokens),
            "read_loop_seconds": read_seconds,
            "records_per_second": len(prepared) / read_seconds
            if read_seconds
            else None,
            "tokens_per_second": tokens / read_seconds if read_seconds else None,
        },
    )


def evaluate_artifacts(
    run: PreparedEvaluationRun, *, ece_bins: int = 15
) -> dict[str, Any]:
    """Return the reusable typed metric report for an evaluation run."""
    return evaluate_decisions(run.rows, ece_bins=ece_bins)


def write_prediction_jsonl(path: str | Path, run: PreparedEvaluationRun) -> Path:
    """Write compact candidate probabilities and immutable canvas provenance."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for prediction in run.predictions:
            stream.write(json.dumps(prediction, sort_keys=True) + "\n")
    return destination


def read_prediction_jsonl(path: str | Path) -> tuple[DecisionEvaluation, ...]:
    """Load compact candidate predictions for paired artifact comparison."""
    source = Path(path)
    rows: list[DecisionEvaluation] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"prediction JSONL has an empty line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"prediction JSONL has invalid JSON at line {line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError("prediction JSONL rows must be objects")
            rows.append(_evaluation_from_prediction(value))
    if not rows:
        raise ValueError("prediction JSONL is empty")
    return tuple(rows)


def write_metrics_json(
    path: str | Path,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    """Write stable JSON metrics with the run's config and model provenance."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"metadata": dict(metadata), **metrics}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def file_sha256(path: str | Path) -> str:
    """Hash a regular file or a deterministic directory tree."""
    source = Path(path)
    if source.is_file():
        return _sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"cannot hash missing path: {source}")
    digest = hashlib.sha256()
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(child.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def prepared_canvas_sha256(run: PreparedEvaluationRun) -> str:
    """Hash the exact prepared canvases, targets, and seeds used for this run."""
    value = [
        {
            key: prediction[key]
            for key in (
                "record_id",
                "question_id",
                "source",
                "allowed_ids",
                "target",
                "seed",
                "steps",
                "canvas",
            )
        }
        for prediction in run.predictions
    ]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prepared_row(
    row: Mapping[str, Any],
    *,
    ordinal_metadata: bool = False,
    codebook: str = "vendored26",
) -> dict[str, Any]:
    canvas = row.get("canvas")
    record = row.get("record")
    source = row.get("source")
    if not isinstance(canvas, DecisionCanvas):
        raise TypeError("prepared evaluation rows require a DecisionCanvas")
    if not isinstance(record, Mapping):
        raise TypeError("prepared evaluation rows require the normalized record")
    if not isinstance(source, str) or not source:
        raise ValueError("prepared evaluation rows require a nonempty source")
    if record.get("source") != source:
        raise ValueError("prepared row source must match its normalized record")
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("prepared normalized records require a nonempty id")
    questions = record.get("questions")
    if not isinstance(questions, Mapping):
        raise TypeError("prepared normalized records require question metadata")
    if len(canvas.question_ids) != len(canvas.targets):
        raise ValueError("canvas questions and targets must align")
    targets = tuple(_target(value) for value in canvas.targets)
    question_types: list[str] = []
    for question_id in canvas.question_ids:
        question = questions.get(question_id)
        if not isinstance(question, Mapping):
            raise ValueError("canvas question IDs must exist in the normalized record")
        question_type = question.get("type")
        if not isinstance(question_type, str) or not question_type:
            raise ValueError("normalized questions require a nonempty type")
        question_types.append(question_type)
    if ordinal_metadata:
        metadata = canvas.ordinal_metadata or ordinal_metadata_for_canvas(
            record, canvas, codebook=codebook
        )
        canvas = replace(canvas, ordinal_metadata=metadata)
    return {
        "canvas": canvas,
        "record_id": record_id,
        "source": source,
        "targets": targets,
        "question_types": tuple(question_types),
    }


def _read_rows(
    prepared: Mapping[str, Any],
    result: DecisionRead,
    seed: int,
    steps: int,
    latency_ms: float,
    *,
    ordinal_metadata: bool,
) -> tuple[tuple[DecisionEvaluation, dict[str, Any]], ...]:
    canvas = prepared["canvas"]
    assert isinstance(canvas, DecisionCanvas)
    question_count = len(canvas.question_ids)
    _validate_read(result, canvas)
    output = []
    for index, question_id in enumerate(canvas.question_ids):
        mask = result.candidate_mask[index]
        candidates = result.allowed_ids[index][mask]
        probabilities = result.restricted_probs[index][mask]
        candidate_logprobs = result.full_vocab_logprobs[index][candidates]
        restricted_logprobs = torch.log_softmax(candidate_logprobs.float(), dim=0)
        allowed_ids = tuple(int(value) for value in candidates.detach().cpu().tolist())
        candidate_probs = tuple(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        candidate_restricted_logprobs = tuple(
            float(value) for value in restricted_logprobs.detach().cpu().tolist()
        )
        expected_probabilities = restricted_logprobs.exp()
        if not torch.allclose(
            probabilities.float(), expected_probabilities, rtol=1e-5, atol=1e-7
        ):
            raise ValueError("reader restricted probabilities disagree with logits")
        if allowed_ids != tuple(canvas.allowed_ids[index]):
            raise ValueError("reader candidate IDs do not match the prepared canvas")
        target = prepared["targets"][index]
        assert isinstance(target, (HardLabel, DistributionLabel, SetLabel))
        semantic_labels = _semantic_labels(canvas, index)
        ordinal = (
            canvas.ordinal_metadata[index]
            if ordinal_metadata and canvas.ordinal_metadata
            else None
        )
        evaluation = DecisionEvaluation(
            record_id=prepared["record_id"],
            question_id=question_id,
            source=prepared["source"],
            question_type=prepared["question_types"][index],
            allowed_ids=allowed_ids,
            probabilities=candidate_probs,
            target=target,
            latency_ms=latency_ms,
            semantic_labels=semantic_labels,
            restricted_logprobs=candidate_restricted_logprobs,
            ordinal_metadata=ordinal,
        )
        prediction = {
            "record_id": evaluation.record_id,
            "question_id": evaluation.question_id,
            "source": evaluation.source,
            "question_type": evaluation.question_type,
            "allowed_ids": list(evaluation.allowed_ids),
            "probabilities": list(evaluation.probabilities),
            "restricted_logprobs": list(evaluation.restricted_logprobs or ()),
            "target": _target_mapping(evaluation.target),
            "semantic_labels": (
                None
                if evaluation.semantic_labels is None
                else list(evaluation.semantic_labels)
            ),
            "latency_ms": evaluation.latency_ms,
            "seed": seed,
            "steps": steps,
            "canvas": _prediction_canvas_mapping(
                canvas, ordinal_metadata=ordinal_metadata
            ),
            "diagnostics": _diagnostics_mapping(result),
        }
        if ordinal_metadata:
            prediction["ordinal_metadata"] = _ordinal_mapping(
                evaluation.ordinal_metadata
            )
        output.append((evaluation, prediction))
    if len(output) != question_count:
        raise AssertionError("reader conversion lost a canvas question")
    return tuple(output)


def _validate_read(result: DecisionRead, canvas: DecisionCanvas) -> None:
    count = len(canvas.question_ids)
    if result.question_ids != tuple(canvas.question_ids):
        raise ValueError("reader question IDs do not match the prepared canvas")
    positions = tuple(
        int(value) for value in result.label_positions.detach().cpu().tolist()
    )
    if positions != tuple(canvas.label_positions):
        raise ValueError("reader label positions do not match the prepared canvas")
    if result.allowed_ids.ndim != 2 or result.candidate_mask.ndim != 2:
        raise ValueError("reader candidate IDs and mask must be matrices")
    if result.full_vocab_logprobs.ndim != 2:
        raise ValueError("reader full-vocabulary logprobs must be a matrix")
    if result.restricted_probs.shape != result.allowed_ids.shape:
        raise ValueError(
            "reader restricted probabilities must align with candidate IDs"
        )
    if result.candidate_mask.shape != result.allowed_ids.shape:
        raise ValueError("reader candidate mask must align with candidate IDs")
    if result.candidate_mask.dtype is not torch.bool:
        raise TypeError("reader candidate mask must be bool")
    if result.allowed_ids.shape[0] != count:
        raise ValueError("reader returned the wrong number of questions")
    if result.full_vocab_logprobs.shape[0] != count:
        raise ValueError("reader full-vocabulary logprobs have the wrong questions")
    tensors = (
        result.allowed_ids,
        result.candidate_mask,
        result.restricted_probs,
        result.full_vocab_logprobs,
    )
    if any(tensor.device != result.allowed_ids.device for tensor in tensors):
        raise ValueError("reader candidate tensors must share a device")
    for index in range(count):
        candidates = result.allowed_ids[index][result.candidate_mask[index]]
        if not candidates.numel():
            raise ValueError("reader returned an empty candidate row")
        logprobs = result.full_vocab_logprobs[index][candidates]
        if not torch.isfinite(logprobs).all():
            raise ValueError("reader omitted a finite logprob for a candidate")


def _validate_prepared_questions(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        canvas = row["canvas"]
        assert isinstance(canvas, DecisionCanvas)
        source = row["source"]
        record_id = row["record_id"]
        assert isinstance(source, str) and isinstance(record_id, str)
        for question_id in canvas.question_ids:
            key = (source, record_id, question_id)
            if key in seen:
                raise ValueError(
                    "prepared evaluation requires unique record and question IDs"
                )
            seen.add(key)


def _semantic_labels(canvas: DecisionCanvas, index: int) -> tuple[str, ...] | None:
    """Return labels only when the prepared canvas explicitly carries semantics."""
    del canvas, index
    return None


def _target(value: object) -> DecisionLabelTarget:
    if isinstance(value, (HardLabel, DistributionLabel, SetLabel)):
        return value
    if isinstance(value, Mapping):
        return label_target_from_mapping(value)
    raise TypeError("canvas targets must be typed labels or normalized mappings")


def _target_mapping(target: DecisionLabelTarget) -> dict[str, Any]:
    if isinstance(target, HardLabel):
        return {"kind": "hard", "gold_idx": target.gold_index}
    if isinstance(target, DistributionLabel):
        result: dict[str, Any] = {"kind": "dist", "probs": list(target.probabilities)}
        if target.candidate_indices is not None:
            result["candidate_indices"] = list(target.candidate_indices)
        if target.candidate_ids is not None:
            result["candidate_ids"] = list(target.candidate_ids)
        if target.other_probability:
            result["other_probability"] = target.other_probability
        return result
    assert isinstance(target, SetLabel)
    return {"kind": "set", "allowed_set": list(target.allowed_indices)}


def _evaluation_from_prediction(value: Mapping[str, Any]) -> DecisionEvaluation:
    try:
        allowed_ids = tuple(value["allowed_ids"])
        probabilities = tuple(value["probabilities"])
        target = _target(value["target"])
        logprob_values = value.get("restricted_logprobs")
        semantic_values = value.get("semantic_labels")
        semantic_labels = None if semantic_values is None else tuple(semantic_values)
        ordinal_metadata = _ordinal_from_mapping(value.get("ordinal_metadata"))
        return DecisionEvaluation(
            record_id=value["record_id"],
            question_id=value["question_id"],
            source=value["source"],
            question_type=value["question_type"],
            allowed_ids=allowed_ids,
            probabilities=probabilities,
            target=target,
            latency_ms=value["latency_ms"],
            semantic_labels=semantic_labels,
            restricted_logprobs=(
                None if logprob_values is None else tuple(logprob_values)
            ),
            ordinal_metadata=ordinal_metadata,
        )
    except KeyError as error:
        raise ValueError(
            f"prediction JSONL row is missing {error.args[0]!r}"
        ) from error


def _ordinal_mapping(
    value: OrdinalMetadata | None,
) -> dict[str, list[str] | list[int]] | None:
    if value is None:
        return None
    return {
        "levels": list(value.levels),
        "source_ids": list(value.source_ids),
        "candidate_ranks": list(value.candidate_ranks),
    }


def _ordinal_metadata_mapping(
    values: Sequence[OrdinalMetadata | None],
) -> list[dict[str, list[str] | list[int]] | None]:
    return [_ordinal_mapping(value) for value in values]


def _ordinal_from_mapping(value: object) -> OrdinalMetadata | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("prediction ordinal_metadata must be an object or null")
    levels = value.get("levels")
    source_ids = value.get("source_ids")
    candidate_ranks = value.get("candidate_ranks")
    if not (
        isinstance(levels, list)
        and isinstance(source_ids, list)
        and isinstance(candidate_ranks, list)
    ):
        raise ValueError("prediction ordinal_metadata fields must be arrays")
    return OrdinalMetadata(
        levels=tuple(levels),
        source_ids=tuple(source_ids),
        candidate_ranks=tuple(candidate_ranks),
    )


def _diagnostics_mapping(result: DecisionRead) -> dict[str, Any] | None:
    diagnostics = result.diagnostics
    if diagnostics is None:
        return None
    return {
        "noise_seed": diagnostics.noise_seed,
        "noise_kind": diagnostics.noise_kind,
        "update_policy": diagnostics.update_policy,
        "steps": diagnostics.steps,
        "forward_count": diagnostics.forward_count,
        "initial_canvas_ids": diagnostics.initial_canvas_ids.detach().cpu().tolist(),
        "final_canvas_ids": diagnostics.final_canvas_ids.detach().cpu().tolist(),
        "slot_init_policy": diagnostics.slot_init_policy,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
