"""Prepared-canvas decision evaluation artifact coverage."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.diffusion_decision.evaluation import (
    evaluate_artifacts,
    evaluate_prepared_rows,
    file_sha256,
    prepared_canvas_sha256,
    read_prediction_jsonl,
    write_metrics_json,
    write_prediction_jsonl,
)
from axolotl.integrations.diffusion_decision.readers.base import DecisionRead
from axolotl.integrations.diffusion_decision.records import DecisionCanvas


def _canvas() -> DecisionCanvas:
    return DecisionCanvas(
        prompt_ids=(1, 2),
        canvas_ids=(3, 4, 5, 6),
        label_positions=(0, 2),
        allowed_ids=((4, 5), (6, 7, 8)),
        question_ids=("q1", "q2"),
        targets=(
            {"kind": "hard", "gold_idx": 0},
            {"kind": "set", "allowed_set": (1, 2)},
        ),
        pinned_mask=(False, False, False, False),
        semantic_mask=(True, True, True, True),
        slot_mask=(False, False, False, False),
        template_length=3,
    )


def _prepared_row() -> dict[str, object]:
    return {
        "canvas": _canvas(),
        "source": "procedural",
        "record": {
            "id": "record-1",
            "source": "procedural",
            "group": "record-1",
            "state": {"example": "state"},
            "questions": {
                "q1": {
                    "type": "choice",
                    "options": [
                        {"name": "first", "description": "first"},
                        {"name": "second", "description": "second"},
                    ],
                },
                "q2": {"type": "noul"},
            },
            "labels": {
                "q1": {"kind": "hard", "gold_idx": 0},
                "q2": {"kind": "hard", "gold_idx": 0},
            },
        },
    }


class _Reader:
    def autocast_context(self, _device):
        return nullcontext()

    def __init__(self, *, mismatched_ids: bool = False) -> None:
        self.calls: list[tuple[DecisionCanvas, int, int]] = []
        self.mismatched_ids = mismatched_ids
        self.attention_backend = "dense"

    def read(self, model, spec, canvas, *, steps, seed, diagnostics):
        del model, spec
        assert diagnostics
        self.calls.append((canvas, steps, seed))
        allowed = torch.tensor([[4, 9 if self.mismatched_ids else 5, 0], [6, 7, 8]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        probabilities = torch.tensor([[0.7, 0.3, 0.0], [0.2, 0.3, 0.5]])
        full = torch.full((2, 16), -torch.inf)
        full[0, 4], full[0, allowed[0, 1]] = math.log(0.7), math.log(0.3)
        full[1, 6], full[1, 7], full[1, 8] = (
            math.log(0.2),
            math.log(0.3),
            math.log(0.5),
        )
        return DecisionRead(
            question_ids=tuple(canvas.question_ids),
            label_positions=torch.tensor((0, 2)),
            allowed_ids=allowed,
            candidate_mask=mask,
            full_vocab_logprobs=full,
            restricted_probs=probabilities,
        )


class _HeldNoiseReader(_Reader):
    def __init__(self) -> None:
        super().__init__()
        self.held: list[bool] = []

    def read(
        self, model, spec, canvas, *, steps, seed, diagnostics, hold_label_noise=False
    ):
        self.held.append(hold_label_noise)
        return super().read(
            model, spec, canvas, steps=steps, seed=seed, diagnostics=diagnostics
        )


class _BatchReader(_HeldNoiseReader):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[tuple[DecisionCanvas, ...], tuple[int, ...]]] = []

    def read_batch(
        self,
        model,
        spec,
        canvases,
        *,
        steps,
        seeds,
        diagnostics,
        hold_label_noise=False,
    ):
        self.batches.append((tuple(canvases), tuple(seeds)))
        return tuple(
            self.read(
                model,
                spec,
                canvas,
                steps=steps,
                seed=seed,
                diagnostics=diagnostics,
                hold_label_noise=hold_label_noise,
            )
            for canvas, seed in zip(canvases, seeds, strict=True)
        )


def test_prepared_evaluation_uses_exact_canvas_targets_and_unpadded_candidates(
    tmp_path,
):
    reader = _Reader()
    ticks = iter((1.0, 1.0125))
    syncs: list[None] = []
    run = evaluate_prepared_rows(
        reader,
        object(),
        object(),
        (_prepared_row(),),
        seed=11,
        warmup=1,
        synchronize=lambda: syncs.append(None),
        clock=lambda: next(ticks),
    )

    assert reader.calls == [(_canvas(), 1, 11), (_canvas(), 1, 11)]
    assert len(syncs) == 2
    first, second = run.rows
    assert first.record_id == "record-1"
    assert first.question_id == "q1"
    assert first.allowed_ids == (4, 5)
    assert first.probabilities == pytest.approx((0.7, 0.3))
    assert first.restricted_logprobs == pytest.approx((math.log(0.7), math.log(0.3)))
    assert second.allowed_ids == (6, 7, 8)
    assert run.predictions[0]["canvas"] == {
        "prompt_ids": [1, 2],
        "canvas_ids": [3, 4, 5, 6],
        "label_positions": [0, 2],
        "allowed_ids": [[4, 5], [6, 7, 8]],
        "pinned_mask": [False, False, False, False],
        "semantic_mask": [True, True, True, True],
        "template_length": 3,
    }
    assert (
        prepared_canvas_sha256(run)
        == "97896b2c9908b5a64b6b1b8f846c01699c1691f65439b671ab67998009f019f3"
    )
    assert run.predictions[0]["seed"] == 11

    report = evaluate_artifacts(run)
    assert report["metrics"]["examples"] == 1
    assert report["metrics"]["questions"] == 2
    prediction_path = write_prediction_jsonl(tmp_path / "predictions.jsonl", run)
    restored = read_prediction_jsonl(prediction_path)
    assert restored == run.rows
    metric_path = write_metrics_json(
        tmp_path / "metrics.json", report, {"seed": 11, "backend": "dense"}
    )
    assert file_sha256(metric_path) == file_sha256(metric_path)


def test_prepared_evaluation_passes_explicit_held_label_noise_for_k_steps():
    reader = _HeldNoiseReader()
    run = evaluate_prepared_rows(
        reader,
        object(),
        object(),
        (_prepared_row(),),
        steps=2,
        hold_label_noise=True,
    )

    assert len(run.rows) == 2
    assert reader.held == [True]
    assert reader.calls == [(_canvas(), 2, 0)]


def test_prepared_evaluation_allows_held_label_noise_for_one_read():
    reader = _HeldNoiseReader()
    run = evaluate_prepared_rows(
        reader,
        object(),
        object(),
        (_prepared_row(),),
        hold_label_noise=True,
    )

    assert len(run.rows) == 2
    assert reader.held == [True]
    assert reader.calls == [(_canvas(), 1, 0)]


def test_prepared_evaluation_batches_reads_without_reordering_seeds_or_records():
    reader = _BatchReader()
    rows = [_prepared_row() for _ in range(3)]
    for index, row in enumerate(rows):
        row["record"] = dict(row["record"], id=f"record-{index}")
    run = evaluate_prepared_rows(
        reader,
        object(),
        object(),
        rows,
        seed=17,
        steps=2,
        hold_label_noise=True,
        batch_size=2,
    )

    assert [seeds for _canvases, seeds in reader.batches] == [(17, 18)]
    assert reader.calls[-1][2] == 19
    assert reader.held == [True, True, True]
    assert [row.record_id for row in run.rows[::2]] == [
        "record-0",
        "record-1",
        "record-2",
    ]
    assert [prediction["seed"] for prediction in run.predictions[::2]] == [17, 18, 19]


def test_prepared_evaluation_rejects_batching_for_single_canvas_reader():
    first, second = _prepared_row(), _prepared_row()
    second["record"] = dict(second["record"], id="record-2")
    with pytest.raises(NotImplementedError, match="does not support batched"):
        evaluate_prepared_rows(
            _Reader(),
            object(),
            object(),
            (first, second),
            batch_size=2,
        )


def test_prepared_evaluation_caps_warmup_and_token_budget_batches():
    reader = _BatchReader()
    rows = [_prepared_row() for _ in range(3)]
    for index, row in enumerate(rows):
        row["record"] = dict(row["record"], id=f"budget-{index}")
    evaluate_prepared_rows(
        reader,
        object(),
        object(),
        rows,
        seed=3,
        warmup=1,
        batch_size=3,
        max_batch_tokens=6,
    )

    assert reader.batches == []
    assert [seed for _canvas, _steps, seed in reader.calls] == [3, 3, 4, 5]


def test_prepared_evaluation_rejects_reader_candidates_that_drift_from_canvas():
    with pytest.raises(ValueError, match="candidate IDs"):
        evaluate_prepared_rows(
            _Reader(mismatched_ids=True), object(), object(), (_prepared_row(),)
        )


def test_prepared_evaluation_rejects_invalid_steps_and_unprepared_rows():
    with pytest.raises(ValueError, match="positive"):
        evaluate_prepared_rows(
            _Reader(), object(), object(), (_prepared_row(),), steps=0
        )
    with pytest.raises(TypeError, match="DecisionCanvas"):
        evaluate_prepared_rows(_Reader(), object(), object(), ({},))


def test_prepared_evaluation_rejects_duplicate_questions_before_reader_calls():
    reader = _Reader()

    with pytest.raises(ValueError, match="unique record and question IDs"):
        evaluate_prepared_rows(
            reader, object(), object(), (_prepared_row(), _prepared_row()), warmup=1
        )

    assert reader.calls == []


def test_prepared_evaluation_logs_periodic_and_final_read_progress():
    rows = []
    for index in range(101):
        canvas = replace(
            _canvas(),
            question_ids=(f"q{index}-one", f"q{index}-two"),
        )
        rows.append(
            {
                "canvas": canvas,
                "source": "procedural",
                "record": {
                    "id": f"record-{index}",
                    "source": "procedural",
                    "questions": {
                        canvas.question_ids[0]: {"type": "choice"},
                        canvas.question_ids[1]: {"type": "set"},
                    },
                },
            }
        )

    messages: list[str] = []

    class MessageHandler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = logging.getLogger("axolotl.integrations.diffusion_decision.evaluation")
    handler = MessageHandler()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        evaluate_prepared_rows(_Reader(), object(), object(), rows)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert "Evaluated decision reads: 100/101" in messages
    assert "Evaluated decision reads: 101/101" in messages


def test_generic_cli_uses_axolotl_runtime_and_writes_paired_artifacts(
    monkeypatch, tmp_path
):
    script = Path(__file__).parents[3] / "scripts/diffusion_lm/decision_evaluate.py"
    spec = importlib.util.spec_from_file_location("decision_evaluate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tmp_path / "config.yml"
    config_path.write_text("base_model: local\n", encoding="utf-8")
    cfg = SimpleNamespace(
        axolotl_config_path=config_path,
        base_model="local",
        revision_of_model="pinned",
        diffusion_decision=SimpleNamespace(
            latent=SimpleNamespace(
                mode="none", num_slots=0, sample_num_slots=False, token_ids=()
            ),
            labels=SimpleNamespace(codebook="expanded52"),
        ),
        seed=5,
        attn_implementation="eager",
        flex_attn_compile_kwargs=None,
        adapter="lora",
        output_dir=tmp_path,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=16, mask_token_id=100, _commit_hash="model"),
        parameters=lambda: iter((torch.nn.Parameter(torch.zeros(())),)),
        eval=lambda: model,
    )
    reader = _Reader()

    class _Tokenizer:
        name_or_path = "tokenizer"
        init_kwargs = {"revision": "pinned"}

        def __len__(self):
            return 16

    loaded_tokenizer = _Tokenizer()

    class _Dataset:
        manifest = {"eval_rows": 1}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return _prepared_row()

    dataset = _Dataset()
    assert module._reader(model, cfg, None).attention_backend == "dense"
    assert (
        module._reader(
            model,
            SimpleNamespace(
                attn_implementation="varlen", flex_attn_compile_kwargs=None
            ),
            None,
        ).attention_backend
        == "varlen"
    )
    monkeypatch.setattr(module, "load_cfg", lambda _path: cfg)

    def _load_runtime(*, cfg, inference):
        assert inference
        assert cfg.adapter is None
        assert cfg.lora_model_dir is None
        return model, loaded_tokenizer, None

    monkeypatch.setattr(
        module,
        "load_model_and_tokenizer",
        _load_runtime,
    )
    monkeypatch.setattr(
        module,
        "load_decision_datasets",
        lambda _cfg, *, tokenizer: (
            pytest.fail("evaluator must pass the loaded tokenizer")
            if tokenizer is not loaded_tokenizer
            else SimpleNamespace(eval_dataset=dataset)
        ),
    )
    monkeypatch.setattr(module, "require_diffusion_spec", lambda _cfg: object())
    monkeypatch.setattr(module, "HFReader", lambda **_kwargs: reader)
    evaluate_prepared_rows = module.evaluate_prepared_rows
    codebooks = []

    def capture_codebook(*args, **kwargs):
        codebooks.append(kwargs["codebook"])
        return evaluate_prepared_rows(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_prepared_rows", capture_codebook)
    audit_path = tmp_path / "diffusion_decision_preparation_audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "preparation_audit_path", lambda _cfg: audit_path)

    output = tmp_path / "output"
    assert (
        module.main(
            [
                str(config_path),
                "--output-dir",
                str(output),
                "--base",
                "--warmup",
                "0",
                "--ordinal-metadata",
            ]
        )
        == 0
    )

    assert (output / "predictions.jsonl").is_file()
    assert (output / "metrics.json").is_file()
    assert reader.calls == [(replace(_canvas(), ordinal_metadata=(None, None)), 1, 5)]
    assert codebooks == ["expanded52"]


@pytest.mark.parametrize("split", ["test", "calibration", "ood"])
def test_cli_explicit_split_uses_only_evaluation_loader_and_records_selection(
    monkeypatch, tmp_path, split
):
    script = Path(__file__).parents[3] / "scripts/diffusion_lm/decision_evaluate.py"
    module_spec = importlib.util.spec_from_file_location(
        f"decision_evaluate_{split}", script
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config_path = tmp_path / "config.yml"
    config_path.write_text("base_model: local\n", encoding="utf-8")
    cfg = SimpleNamespace(
        axolotl_config_path=config_path,
        base_model="local",
        revision_of_model="pinned",
        diffusion_decision=SimpleNamespace(
            latent=SimpleNamespace(
                mode="none", num_slots=0, sample_num_slots=False, token_ids=()
            ),
        ),
        seed=5,
        attn_implementation="eager",
        flex_attn_compile_kwargs=None,
        adapter="lora",
        output_dir=tmp_path,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=16, mask_token_id=100, _commit_hash="model"),
        parameters=lambda: iter((torch.nn.Parameter(torch.zeros(())),)),
        eval=lambda: model,
    )

    class _Tokenizer:
        name_or_path = "tokenizer"
        init_kwargs = {"revision": "pinned"}

        def __len__(self):
            return 16

    audit_path = tmp_path / f"{split}-audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")

    class _Dataset:
        manifest = {
            "selected_split": split,
            "selected_source_inputs": [
                {"collection": "test_datasets", "path": f"{split}-only"}
            ],
            "input_rows": 1,
            "grouped_rows": 1,
            "eval_rows": 1,
            "eval_sources": [split],
            "canvas_too_long": {"eval": 0},
            "budget_drops": {"eval": {"logical": 0, "physical": 0}},
            "preparation_audit": str(audit_path),
        }

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return _prepared_row()

    calls = []
    monkeypatch.setattr(module, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        module,
        "load_model_and_tokenizer",
        lambda *, cfg, inference: (model, _Tokenizer(), None),
    )
    monkeypatch.setattr(
        module,
        "load_decision_datasets",
        lambda *_args, **_kwargs: pytest.fail("default loader must not read train/dev"),
    )
    monkeypatch.setattr(
        module,
        "load_decision_evaluation_dataset",
        lambda passed_cfg, tokenizer, passed_split: (
            calls.append((passed_cfg, tokenizer, passed_split)) or _Dataset()
        ),
    )
    monkeypatch.setattr(module, "require_diffusion_spec", lambda _cfg: object())
    monkeypatch.setattr(module, "HFReader", lambda **_kwargs: _Reader())

    output = tmp_path / split
    assert (
        module.main(
            [
                str(config_path),
                "--output-dir",
                str(output),
                "--base",
                "--split",
                split,
                "--warmup",
                "0",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0][2] == split
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    selection = metrics["metadata"]["evaluation_selection"]
    assert selection["selected_split"] == split
    assert selection["source_inputs"] == _Dataset.manifest["selected_source_inputs"]
    assert selection["source_counts"] == {
        "input_rows": 1,
        "grouped_rows": 1,
        "eval_rows": 1,
    }
    assert selection["drops"] == {
        "canvas_too_long": {"eval": 0},
        "budget_drops": {"eval": {"logical": 0, "physical": 0}},
    }


def test_adapter_payload_fingerprint_excludes_runtime_logs_and_checkpoints(tmp_path):
    from axolotl.integrations.diffusion_decision.evaluation import (
        ADAPTER_FINGERPRINT_VERSION,
        adapter_payload_fingerprint,
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "diffusion_decision_manifest.json").write_text("{}\n", encoding="utf-8")
    (adapter / "debug.log").write_text("first\n", encoding="utf-8")
    checkpoint = adapter / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("different\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    nested = adapter / "outputs" / "nested-adapter"
    nested.mkdir(parents=True)
    (nested / "adapter_model.safetensors").write_bytes(b"nested")

    first = adapter_payload_fingerprint(adapter)
    (adapter / "debug.log").write_text("rewritten\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"rewritten checkpoint")
    (nested / "adapter_model.safetensors").write_bytes(b"rewritten nested")
    second = adapter_payload_fingerprint(adapter)

    assert first == second
    assert first["version"] == ADAPTER_FINGERPRINT_VERSION
    assert first["files"] == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "diffusion_decision_manifest.json",
    ]


def test_adapter_payload_fingerprint_hashes_index_declared_shards_and_manifest(
    tmp_path,
):
    from axolotl.integrations.diffusion_decision.evaluation import (
        adapter_payload_fingerprint,
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "diffusion_decision_manifest.json").write_text("one\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}',
        encoding="utf-8",
    )
    (adapter / "part-1.safetensors").write_bytes(b"one")
    (adapter / "part-2.safetensors").write_bytes(b"two")

    first = adapter_payload_fingerprint(adapter)
    (adapter / "part-2.safetensors").write_bytes(b"changed")
    second = adapter_payload_fingerprint(adapter)
    (adapter / "diffusion_decision_manifest.json").write_text(
        "changed\n", encoding="utf-8"
    )
    third = adapter_payload_fingerprint(adapter)

    assert first["sha256"] != second["sha256"] != third["sha256"]
    assert second["files"] == [
        "adapter_config.json",
        "adapter_model.safetensors.index.json",
        "diffusion_decision_manifest.json",
        "part-1.safetensors",
        "part-2.safetensors",
    ]


@pytest.mark.parametrize(
    ("weight_map", "message"),
    [
        ({"a": "missing.safetensors"}, "missing shard"),
        ({"a": "../outside.safetensors"}, "unsafe shard path"),
        ({"a": "/absolute.safetensors"}, "unsafe shard path"),
    ],
)
def test_adapter_payload_fingerprint_rejects_invalid_index_shards(
    tmp_path, weight_map, message
):
    from axolotl.integrations.diffusion_decision.evaluation import (
        adapter_payload_fingerprint,
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        adapter_payload_fingerprint(adapter)


def test_adapter_payload_fingerprint_requires_config_and_weights(tmp_path):
    from axolotl.integrations.diffusion_decision.evaluation import (
        adapter_payload_fingerprint,
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="adapter_config"):
        adapter_payload_fingerprint(adapter)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root adapter model weight"):
        adapter_payload_fingerprint(adapter)


def test_adapter_payload_fingerprint_rejects_mixed_direct_and_indexed_weights(tmp_path):
    from axolotl.integrations.diffusion_decision.evaluation import (
        adapter_payload_fingerprint,
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"direct")
    (adapter / "adapter_model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "part-1.safetensors"}}', encoding="utf-8"
    )
    (adapter / "part-1.safetensors").write_bytes(b"shard")

    with pytest.raises(ValueError, match="mixed direct and indexed"):
        adapter_payload_fingerprint(adapter)
