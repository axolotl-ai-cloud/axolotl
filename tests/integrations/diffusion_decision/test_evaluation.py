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
    ADAPTER_FINGERPRINT_VERSION,
    adapter_payload_fingerprint,
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

from tests.integrations.diffusion_decision.helpers import make_canvas, make_record

CLI_SCRIPT = Path(__file__).parents[3] / "scripts/diffusion_lm/decision_evaluate.py"


def _canvas() -> DecisionCanvas:
    return make_canvas(
        (1, 2),
        (3, 4, 5, 6),
        (0, 2),
        allowed_ids=((4, 5), (6, 7, 8)),
        question_ids=("q1", "q2"),
        targets=(
            {"kind": "hard", "gold_idx": 0},
            {"kind": "set", "allowed_set": (1, 2)},
        ),
        template_length=3,
    )


def _prepared_row(record_id: str = "record-1") -> dict[str, object]:
    record = make_record(
        record_id,
        source="procedural",
        group="record-1",
        state={"example": "state"},
        questions={
            "q1": {
                "type": "choice",
                "options": [
                    {"name": "first", "description": "first"},
                    {"name": "second", "description": "second"},
                ],
            },
            "q2": {"type": "noul"},
        },
        labels={
            "q1": {"kind": "hard", "gold_idx": 0},
            "q2": {"kind": "hard", "gold_idx": 0},
        },
    )
    return {"canvas": _canvas(), "source": "procedural", "record": record}


def _evaluate(reader, rows, **kwargs):
    return evaluate_prepared_rows(reader, object(), object(), rows, **kwargs)


class _Reader:
    def __init__(self, *, mismatched_ids: bool = False) -> None:
        self.calls: list[tuple[DecisionCanvas, int, int]] = []
        self.held: list[bool] = []
        self.mismatched_ids = mismatched_ids
        self.attention_backend = "dense"

    def autocast_context(self, _device):
        return nullcontext()

    def read(
        self, model, spec, canvas, *, steps, seed, diagnostics, hold_label_noise=False
    ):
        del model, spec
        assert diagnostics
        self.calls.append((canvas, steps, seed))
        self.held.append(hold_label_noise)
        second = 9 if self.mismatched_ids else 5
        full = torch.full((2, 16), -torch.inf)
        full[0, 4], full[0, second] = math.log(0.7), math.log(0.3)
        full[1, 6:9] = torch.tensor((0.2, 0.3, 0.5)).log()
        return DecisionRead(
            question_ids=tuple(canvas.question_ids),
            label_positions=torch.tensor((0, 2)),
            allowed_ids=torch.tensor([[4, second, 0], [6, 7, 8]]),
            candidate_mask=torch.tensor([[True, True, False], [True, True, True]]),
            full_vocab_logprobs=full,
            restricted_probs=torch.tensor([[0.7, 0.3, 0.0], [0.2, 0.3, 0.5]]),
        )


class _BatchReader(_Reader):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[tuple[DecisionCanvas, ...], tuple[int, ...]]] = []

    def read_batch(self, model, spec, canvases, *, seeds, **kwargs):
        self.batches.append((tuple(canvases), tuple(seeds)))
        return tuple(
            self.read(model, spec, canvas, seed=seed, **kwargs)
            for canvas, seed in zip(canvases, seeds, strict=True)
        )


def test_prepared_evaluation_uses_exact_canvas_targets_and_unpadded_candidates(
    tmp_path,
):
    reader = _Reader()
    ticks = iter((1.0, 1.0125))
    syncs: list[None] = []
    run = _evaluate(
        reader,
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


@pytest.mark.parametrize("steps", [1, 2], ids=["one_read", "k_steps"])
def test_prepared_evaluation_passes_explicit_held_label_noise(steps):
    reader = _Reader()
    run = _evaluate(reader, (_prepared_row(),), steps=steps, hold_label_noise=True)

    assert len(run.rows) == 2
    assert reader.held == [True]
    assert reader.calls == [(_canvas(), steps, 0)]


def test_prepared_evaluation_batches_reads_without_reordering_seeds_or_records():
    reader = _BatchReader()
    rows = [_prepared_row(f"record-{index}") for index in range(3)]
    run = _evaluate(reader, rows, seed=17, steps=2, hold_label_noise=True, batch_size=2)

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
    rows = (_prepared_row(), _prepared_row("record-2"))
    with pytest.raises(NotImplementedError, match="does not support batched"):
        _evaluate(_Reader(), rows, batch_size=2)


def test_prepared_evaluation_caps_warmup_and_token_budget_batches():
    reader = _BatchReader()
    rows = [_prepared_row(f"budget-{index}") for index in range(3)]
    _evaluate(reader, rows, seed=3, warmup=1, batch_size=3, max_batch_tokens=6)

    assert reader.batches == []
    assert [seed for _canvas, _steps, seed in reader.calls] == [3, 3, 4, 5]


@pytest.mark.parametrize(
    ("mismatched_ids", "rows", "kwargs", "error", "message"),
    [
        (True, (_prepared_row(),), {}, ValueError, "candidate IDs"),
        (False, (_prepared_row(),), {"steps": 0}, ValueError, "positive"),
        (False, ({},), {}, TypeError, "DecisionCanvas"),
    ],
    ids=["drifted_candidates", "zero_steps", "unprepared_row"],
)
def test_prepared_evaluation_rejects_invalid_reads(
    mismatched_ids, rows, kwargs, error, message
):
    with pytest.raises(error, match=message):
        _evaluate(_Reader(mismatched_ids=mismatched_ids), rows, **kwargs)


def test_prepared_evaluation_rejects_duplicate_questions_before_reader_calls():
    reader = _Reader()

    with pytest.raises(ValueError, match="unique record and question IDs"):
        _evaluate(reader, (_prepared_row(), _prepared_row()), warmup=1)

    assert reader.calls == []


def test_prepared_evaluation_logs_periodic_and_final_read_progress(caplog):
    rows = []
    for index in range(101):
        canvas = replace(_canvas(), question_ids=(f"q{index}-one", f"q{index}-two"))
        record = {
            "id": f"record-{index}",
            "source": "procedural",
            "questions": {
                canvas.question_ids[0]: {"type": "choice"},
                canvas.question_ids[1]: {"type": "set"},
            },
        }
        rows.append({"canvas": canvas, "source": "procedural", "record": record})

    with caplog.at_level(
        logging.INFO, logger="axolotl.integrations.diffusion_decision.evaluation"
    ):
        _evaluate(_Reader(), rows)

    assert "Evaluated decision reads: 100/101" in caplog.messages
    assert "Evaluated decision reads: 101/101" in caplog.messages


class _Tokenizer:
    name_or_path = "tokenizer"
    init_kwargs = {"revision": "pinned"}

    def __len__(self):
        return 16


class _Dataset(list):
    manifest: dict[str, object]


def _prepared_dataset(manifest: dict[str, object]) -> _Dataset:
    dataset = _Dataset([_prepared_row()])
    dataset.manifest = manifest
    return dataset


def _cli_env(monkeypatch, tmp_path, name, **diffusion_decision):
    module_spec = importlib.util.spec_from_file_location(name, CLI_SCRIPT)
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
            **diffusion_decision,
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
    tokenizer = _Tokenizer()
    reader = _Reader()

    def _load_runtime(*, cfg, inference):
        assert inference
        assert cfg.adapter is None
        assert cfg.lora_model_dir is None
        return model, tokenizer, None

    monkeypatch.setattr(module, "load_cfg", lambda _path: cfg)
    monkeypatch.setattr(module, "load_model_and_tokenizer", _load_runtime)
    monkeypatch.setattr(module, "require_diffusion_spec", lambda _cfg: object())

    def main(output: Path, *extra: str) -> int:
        base = ["--output-dir", str(output), "--base", "--warmup", "0"]
        return module.main([str(config_path), *base, *extra])

    return SimpleNamespace(
        module=module,
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        reader=reader,
        main=main,
    )


def test_generic_cli_uses_axolotl_runtime_and_writes_paired_artifacts(
    monkeypatch, tmp_path
):
    env = _cli_env(
        monkeypatch,
        tmp_path,
        "decision_evaluate",
        labels=SimpleNamespace(codebook="expanded52"),
    )
    module = env.module
    dataset = _prepared_dataset({"eval_rows": 1})
    assert module._reader(env.model, env.cfg, None).attention_backend == "dense"
    varlen_cfg = SimpleNamespace(
        attn_implementation="varlen", flex_attn_compile_kwargs=None
    )
    assert module._reader(env.model, varlen_cfg, None).attention_backend == "varlen"
    monkeypatch.setattr(module, "HFReader", lambda **_kwargs: env.reader)
    monkeypatch.setattr(
        module,
        "load_decision_datasets",
        lambda _cfg, *, tokenizer: (
            pytest.fail("evaluator must pass the loaded tokenizer")
            if tokenizer is not env.tokenizer
            else SimpleNamespace(eval_dataset=dataset)
        ),
    )
    evaluate_rows = module.evaluate_prepared_rows
    codebooks = []

    def capture_codebook(*args, **kwargs):
        codebooks.append(kwargs["codebook"])
        return evaluate_rows(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_prepared_rows", capture_codebook)
    audit_path = tmp_path / "diffusion_decision_preparation_audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "preparation_audit_path", lambda _cfg: audit_path)

    output = tmp_path / "output"
    assert env.main(output, "--ordinal-metadata") == 0

    assert (output / "predictions.jsonl").is_file()
    assert (output / "metrics.json").is_file()
    assert env.reader.calls == [
        (replace(_canvas(), ordinal_metadata=(None, None)), 1, 5)
    ]
    assert codebooks == ["expanded52"]


@pytest.mark.parametrize("split", ["test", "calibration", "ood"])
def test_cli_explicit_split_uses_only_evaluation_loader_and_records_selection(
    monkeypatch, tmp_path, split
):
    env = _cli_env(monkeypatch, tmp_path, f"decision_evaluate_{split}")
    audit_path = tmp_path / f"{split}-audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
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
    calls = []
    monkeypatch.setattr(env.module, "HFReader", lambda **_kwargs: env.reader)
    monkeypatch.setattr(
        env.module,
        "load_decision_datasets",
        lambda *_args, **_kwargs: pytest.fail("default loader must not read train/dev"),
    )
    monkeypatch.setattr(
        env.module,
        "load_decision_evaluation_dataset",
        lambda passed_cfg, tokenizer, passed_split: (
            calls.append((passed_cfg, tokenizer, passed_split))
            or _prepared_dataset(manifest)
        ),
    )

    output = tmp_path / split
    assert env.main(output, "--split", split) == 0
    assert len(calls) == 1
    assert calls[0][2] == split
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    selection = metrics["metadata"]["evaluation_selection"]
    assert selection["selected_split"] == split
    assert selection["source_inputs"] == manifest["selected_source_inputs"]
    assert selection["source_counts"] == {
        "input_rows": 1,
        "grouped_rows": 1,
        "eval_rows": 1,
    }
    assert selection["drops"] == {
        "canvas_too_long": {"eval": 0},
        "budget_drops": {"eval": {"logical": 0, "physical": 0}},
    }


def _write_adapter(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(exist_ok=True)
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content, encoding="utf-8")
    return root


def test_adapter_payload_fingerprint_excludes_runtime_logs_and_checkpoints(tmp_path):
    adapter = _write_adapter(
        tmp_path / "adapter",
        {
            "adapter_config.json": '{"r": 8}\n',
            "adapter_model.safetensors": "weights",
            "diffusion_decision_manifest.json": "{}\n",
            "debug.log": "first\n",
            "checkpoint-50/adapter_config.json": "different\n",
            "checkpoint-50/adapter_model.safetensors": "checkpoint",
            "outputs/nested-adapter/adapter_model.safetensors": "nested",
        },
    )

    first = adapter_payload_fingerprint(adapter)
    _write_adapter(
        adapter,
        {
            "debug.log": "rewritten\n",
            "checkpoint-50/adapter_model.safetensors": "rewritten checkpoint",
            "outputs/nested-adapter/adapter_model.safetensors": "rewritten nested",
        },
    )
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
    adapter = _write_adapter(
        tmp_path / "adapter",
        {
            "adapter_config.json": "{}\n",
            "diffusion_decision_manifest.json": "one\n",
            "adapter_model.safetensors.index.json": json.dumps(
                {"weight_map": {"a": "part-1.safetensors", "": "part-2.safetensors"}}
            ),
            "part-1.safetensors": "one",
            "part-2.safetensors": "two",
        },
    )

    first = adapter_payload_fingerprint(adapter)
    _write_adapter(adapter, {"part-2.safetensors": "changed"})
    second = adapter_payload_fingerprint(adapter)
    _write_adapter(adapter, {"diffusion_decision_manifest.json": "changed\n"})
    third = adapter_payload_fingerprint(adapter)

    assert first["sha256"] != second["sha256"] != third["sha256"]
    assert second["files"] == [
        "adapter_config.json",
        "adapter_model.safetensors.index.json",
        "diffusion_decision_manifest.json",
        "part-1.safetensors",
        "part-2.safetensors",
    ]


def _indexed(shard: str, **extra: str) -> dict[str, str]:
    index = json.dumps({"weight_map": {"a": shard}})
    files = {
        "adapter_config.json": "{}\n",
        "adapter_model.safetensors.index.json": index,
    }
    return {**files, **extra}


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({}, "adapter_config"),
        ({"adapter_config.json": "{}\n"}, "root adapter model weight"),
        (
            _indexed(
                "part-1.safetensors",
                **{
                    "adapter_model.safetensors": "direct",
                    "part-1.safetensors": "shard",
                },
            ),
            "mixed direct and indexed",
        ),
        (_indexed("missing.safetensors"), "missing shard"),
        (_indexed("../outside.safetensors"), "unsafe shard path"),
        (_indexed("/absolute.safetensors"), "unsafe shard path"),
    ],
    ids=[
        "missing_config",
        "missing_weights",
        "mixed_direct_and_indexed",
        "missing_shard",
        "relative_escape_shard",
        "absolute_shard",
    ],
)
def test_adapter_payload_fingerprint_rejects_invalid_layouts(tmp_path, files, message):
    adapter = _write_adapter(tmp_path / "adapter", files)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        adapter_payload_fingerprint(adapter)
