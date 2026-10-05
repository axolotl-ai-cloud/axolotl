"""Ordinal score provenance and ranked probability score regressions."""

from __future__ import annotations

import math

import pytest
import torch

from axolotl.integrations.diffusion_decision.adapters import normalize_record
from axolotl.integrations.diffusion_decision.evaluation import (
    _prepared_row,
    evaluate_prepared_rows,
    prepared_canvas_sha256,
    prepared_rows_canvas_sha256,
    read_prediction_jsonl,
    write_prediction_jsonl,
)
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
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.readers.base import DecisionRead
from axolotl.integrations.diffusion_decision.records import (
    DecisionCanvas,
    OrdinalMetadata,
)
from axolotl.integrations.diffusion_decision.template import SchemaError


def _ordinal(*ranks: int) -> OrdinalMetadata:
    return OrdinalMetadata(
        levels=tuple(f"level-{index}" for index in range(len(ranks))),
        source_ids=tuple(str(index) for index in range(len(ranks))),
        candidate_ranks=ranks,
    )


def test_ordinal_metadata_rebuild_uses_expanded_codebook_for_28_choice_record():
    names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZab")
    record = {
        "id": "go-emotions-28",
        "source": "score_eval",
        "group": "go-emotions-28",
        "state": {"text": "example"},
        "questions": {
            "q1": {
                "type": "choice",
                "options": [{"name": name, "description": name} for name in names],
            }
        },
        "labels": {"q1": {"kind": "hard", "gold_idx": 0}},
    }
    canvas = DecisionCanvas(
        prompt_ids=(1,),
        canvas_ids=(0,) * 128,
        label_positions=(1,),
        allowed_ids=(tuple(range(28)),),
        question_ids=("q1",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(False,) * 128,
        semantic_mask=(False,) * 128,
        slot_mask=(False,) * 128,
        template_length=2,
    )
    row = {"canvas": canvas, "record": record, "source": "score_eval"}

    with pytest.raises(SchemaError, match="at most 26"):
        _prepared_row(row, ordinal_metadata=True)
    prepared = _prepared_row(row, ordinal_metadata=True, codebook="expanded52")
    assert prepared["canvas"].ordinal_metadata == (None,)


def _score_row(
    record_id: str,
    probabilities: tuple[float, ...],
    target,
    *,
    ranks: tuple[int, ...] = (0, 1, 2),
) -> DecisionEvaluation:
    return DecisionEvaluation(
        record_id=record_id,
        question_id=f"q-{record_id}-{type(target).__name__}-{ranks}",
        source="score_eval",
        question_type="score",
        allowed_ids=(41, 73, 19),
        probabilities=probabilities,
        target=target,
        latency_ms=1.0,
        ordinal_metadata=_ordinal(*ranks),
    )


def test_normalized_rps_for_hard_and_distribution_targets_is_record_balanced():
    hard = _score_row("shared", (0.2, 0.5, 0.3), HardLabel(2))
    distribution = _score_row(
        "shared", (0.2, 0.5, 0.3), DistributionLabel((0.1, 0.7, 0.2))
    )
    exact = _score_row("other", (0.0, 0.0, 1.0), HardLabel(2))

    report = evaluate_decisions((hard, distribution, exact))

    hard_rps = (0.2**2 + 0.7**2) / 2
    distribution_rps = (0.1**2 + (-0.1) ** 2) / 2
    assert report["metrics"]["rps"] == pytest.approx(
        ((hard_rps + distribution_rps) / 2 + 0.0) / 2
    )
    assert report["definitions"]["rps"].startswith("score-only")
    assert report["metrics"]["rps_eligible_questions"] == 3
    assert report["metrics"]["rps_eligible_records"] == 2


def test_rps_is_invariant_to_candidate_permutation_when_ranks_remap():
    source_order = _score_row(
        "source", (0.2, 0.5, 0.3), DistributionLabel((0.1, 0.7, 0.2))
    )
    permuted = DecisionEvaluation(
        record_id="permuted",
        question_id="q",
        source="score_eval",
        question_type="score",
        allowed_ids=(19, 41, 73),
        probabilities=(0.3, 0.2, 0.5),
        target=DistributionLabel((0.2, 0.1, 0.7)),
        latency_ms=1.0,
        ordinal_metadata=_ordinal(2, 0, 1),
    )

    first = evaluate_decisions((source_order,))["metrics"]["rps"]
    second = evaluate_decisions((permuted,))["metrics"]["rps"]

    assert first == pytest.approx(second)


@pytest.mark.parametrize(
    "ranks, levels, source_ids, message",
    [
        ((0, 0, 2), ("low", "mid", "high"), ("0", "1", "2"), "zero-based bijection"),
        ((0, 1), ("low", "mid"), ("0", "1"), "align"),
        ((0, 1, True), ("low", "mid", "high"), ("0", "1", "2"), "integers"),
        ((0, 1, 2), ("low", "mid", "high"), ("0", "0", "2"), "source IDs unique"),
    ],
)
def test_rps_rejects_invalid_ordinal_contracts(ranks, levels, source_ids, message):
    row = _score_row("bad", (0.2, 0.5, 0.3), HardLabel(1))

    with pytest.raises(ValueError, match=message):
        metadata = OrdinalMetadata(levels, source_ids, ranks)
        evaluate_decisions(
            (row.__class__(**{**row.__dict__, "ordinal_metadata": metadata}),)
        )


def test_paired_contract_rejects_missing_or_changed_ordinal_metadata():
    current = _score_row("record", (0.2, 0.5, 0.3), HardLabel(1))
    missing = DecisionEvaluation(
        record_id=current.record_id,
        question_id=current.question_id,
        source=current.source,
        question_type=current.question_type,
        allowed_ids=current.allowed_ids,
        probabilities=current.probabilities,
        target=current.target,
        latency_ms=current.latency_ms,
    )

    with pytest.raises(ValueError, match="question contracts"):
        paired_bootstrap((missing,), (current,), metric="rps", seed=0, draws=1)


def test_rps_coverage_is_zero_when_ordinal_metadata_is_unavailable():
    row = DecisionEvaluation(
        record_id="choice",
        question_id="q",
        source="source",
        question_type="choice",
        allowed_ids=(1, 2, 3),
        probabilities=(0.2, 0.5, 0.3),
        target=HardLabel(1),
        latency_ms=1.0,
    )

    metrics = evaluate_decisions((row,))["metrics"]

    assert metrics["rps"] is None
    assert metrics["rps_eligible_questions"] == 0
    assert metrics["rps_eligible_records"] == 0


def test_rps_rejects_single_candidate_ordinal_contract():
    row = DecisionEvaluation(
        record_id="single",
        question_id="q",
        source="source",
        question_type="score",
        allowed_ids=(1,),
        probabilities=(1.0,),
        target=HardLabel(0),
        latency_ms=1.0,
        ordinal_metadata=OrdinalMetadata(
            levels=("only",), source_ids=("0",), candidate_ranks=(0,)
        ),
    )

    with pytest.raises(ValueError, match="at least two candidates"):
        evaluate_decisions((row,))


def test_rps_requires_score_hard_or_distribution_with_metadata():
    choice = DecisionEvaluation(
        record_id="choice",
        question_id="q",
        source="source",
        question_type="choice",
        allowed_ids=(1, 2, 3),
        probabilities=(0.2, 0.5, 0.3),
        target=HardLabel(1),
        latency_ms=1.0,
    )
    score_set = _score_row("set", (0.2, 0.5, 0.3), SetLabel((0, 1)))
    missing = DecisionEvaluation(
        record_id="missing",
        question_id="q",
        source="source",
        question_type="score",
        allowed_ids=(1, 2, 3),
        probabilities=(0.2, 0.5, 0.3),
        target=HardLabel(1),
        latency_ms=1.0,
    )

    assert evaluate_decisions((choice,))["metrics"]["rps"] is None
    assert evaluate_decisions((score_set,))["metrics"]["rps"] is None
    assert evaluate_decisions((missing,))["metrics"]["rps"] is None


class _Reader:
    attention_backend = "dense"

    def read(self, model, spec, canvas, *, steps, seed, diagnostics):
        del model, spec, steps, seed, diagnostics
        allowed = torch.tensor([[31, 32, 33]])
        mask = torch.tensor([[True, True, True]])
        probabilities = torch.tensor([[0.2, 0.5, 0.3]])
        full = torch.full((1, 64), -torch.inf)
        full[0, 31], full[0, 32], full[0, 33] = (
            math.log(0.2),
            math.log(0.5),
            math.log(0.3),
        )
        return DecisionRead(
            question_ids=tuple(canvas.question_ids),
            label_positions=torch.tensor(canvas.label_positions),
            allowed_ids=allowed,
            candidate_mask=mask,
            full_vocab_logprobs=full,
            restricted_probs=probabilities,
        )


def test_soft_score_normalizes_to_opt_in_read_json_roundtrip(monkeypatch, tmp_path):
    import axolotl.integrations.diffusion_decision.preprocessing as preprocessing

    normalized = normalize_record(
        "jsonl",
        {
            "id": "soft-score",
            "source": "score_eval",
            "group": "soft-score",
            "state": {"evidence": "text"},
            "questions": {
                "q1": {
                    "type": "score",
                    "instructions": "",
                    "levels": ["low", "medium", "high"],
                }
            },
            "labels": {"q1": {"kind": "dist", "probs": [0.2, 0.5, 0.3]}},
        },
        training=False,
    )

    monkeypatch.setattr(
        preprocessing,
        "resolve_template",
        lambda *_args, **_kwargs: (
            [9, 31, 11],
            [{"pos": 1, "label_ids": [31, 32, 33]}],
        ),
    )
    canvas = build_decision_canvas(
        object(),
        normalized,
        prompt_ids=(1,),
        scaffold_ids=(),
        turn_close_id=2,
        pad_id=0,
        vocab_size=64,
        width=8,
        include_ordinal_metadata=True,
    )
    assert canvas.ordinal_metadata == (
        OrdinalMetadata(
            levels=("low", "medium", "high"),
            source_ids=("0", "1", "2"),
            candidate_ranks=(0, 1, 2),
        ),
    )

    prepared = ({"canvas": canvas, "record": normalized, "source": "score_eval"},)
    default = evaluate_prepared_rows(_Reader(), object(), object(), prepared)
    assert "ordinal_metadata" not in default.predictions[0]
    assert "ordinal_metadata" not in default.predictions[0]["canvas"]

    run = evaluate_prepared_rows(
        _Reader(), object(), object(), prepared, ordinal_metadata=True
    )
    assert run.rows[0].ordinal_metadata == canvas.ordinal_metadata[0]
    assert run.predictions[0]["ordinal_metadata"] == {
        "levels": ["low", "medium", "high"],
        "source_ids": ["0", "1", "2"],
        "candidate_ranks": [0, 1, 2],
    }
    assert run.predictions[0]["canvas"]["ordinal_metadata"] == [
        run.predictions[0]["ordinal_metadata"]
    ]
    assert prepared_canvas_sha256(default) != prepared_canvas_sha256(run)
    assert prepared_rows_canvas_sha256(prepared) != prepared_rows_canvas_sha256(
        prepared, ordinal_metadata=True
    )

    output = write_prediction_jsonl(tmp_path / "predictions.jsonl", run)
    assert read_prediction_jsonl(output) == run.rows
