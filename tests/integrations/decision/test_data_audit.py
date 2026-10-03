"""Tests for compact decision-dataset preparation provenance."""

import json

from axolotl.integrations.decision.data_audit import (
    PREPARATION_AUDIT_FILENAME,
    PREPARATION_AUDIT_SCHEMA_VERSION,
    build_preparation_audit,
    preparation_audit_path,
    write_preparation_audit,
)
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.utils.dict import DictDefault


def _canvas(prompt_length, width=4):
    return DecisionCanvas(
        prompt_ids=tuple(range(prompt_length)),
        canvas_ids=tuple(range(width)),
        label_positions=(1,),
        allowed_ids=((4, 5),),
        question_ids=("q1",),
        targets=(0,),
        pinned_mask=(True,) * width,
        semantic_mask=(True,) * width,
        slot_mask=(False,) * width,
        template_length=1,
    )


def _row(source, prompt_length, logical_length, question_types):
    return {
        "source": source,
        "canvas": _canvas(prompt_length),
        "length": logical_length,
        "record": {
            "id": f"{source}-{prompt_length}",
            "questions": {
                f"q{index}": {"type": question_type}
                for index, question_type in enumerate(question_types, start=1)
            },
        },
    }


def _cfg(tmp_path):
    return DictDefault(
        {
            "base_model": "nvidia/Nemotron-Labs-Diffusion-3B",
            "revision_of_model": "0d51902da1f8869f83413ce642fab402fa5641e0",
            "seed": 42,
            "credential_that_must_not_be_persisted": "not-a-provenance-field",
            "output_dir": str(tmp_path / "output"),
            "dataset_prepared_path": str(tmp_path / "prepared"),
            "datasets": [
                {
                    "path": "json",
                    "data_files": {"train": "/data/train.jsonl"},
                    "revision": "data-r1",
                    "split": "train",
                    "type": "decision.nimble",
                },
                {"path": "ignored", "type": "chat_template"},
            ],
            "test_datasets": [
                {
                    "path": "org/eval-set",
                    "name": "heldout",
                    "revision": "data-r2",
                    "split": "test",
                    "type": "decision.open_jev",
                }
            ],
        }
    )


def test_audit_persists_compact_manifest_and_per_source_distributions(tmp_path):
    cfg = _cfg(tmp_path)
    manifest = {
        "train_rows": 3,
        "mixture_seed": 42,
        "stratified_epoch_batches": (("nimble", "open_jev"), ("nimble",)),
    }
    train_rows = [
        _row("nimble", 3, 7, ("choice",)),
        _row("nimble", 8, 12, ("noul", "score")),
        _row("open_jev", 5, 9, ("choice",)),
    ]
    eval_rows = [_row("open_jev", 6, 10, ("choice", "choice"))]

    path = write_preparation_audit(cfg, train_rows, eval_rows, manifest)

    assert path == preparation_audit_path(cfg)
    assert path.name == PREPARATION_AUDIT_FILENAME
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PREPARATION_AUDIT_SCHEMA_VERSION
    assert "config_sha256" not in payload
    assert "credential_that_must_not_be_persisted" not in payload["config"]
    assert payload["source_inputs"] == [
        {
            "adapter": "decision.nimble",
            "collection": "datasets",
            "data_files": {"train": "/data/train.jsonl"},
            "index": 0,
            "name": None,
            "path": "json",
            "revision": "data-r1",
            "split": "train",
        },
        {
            "adapter": "decision.open_jev",
            "collection": "test_datasets",
            "index": 0,
            "name": "heldout",
            "path": "org/eval-set",
            "revision": "data-r2",
            "split": "test",
        },
    ]
    assert "stratified_epoch_batches" not in payload["preparation_manifest"]
    assert payload["preparation_manifest"]["stratified_epoch_batches_summary"] == {
        "batches": 2,
        "examples": 3,
    }
    assert payload["splits"]["train"]["rows"] == 3
    assert payload["splits"]["train"]["questions"] == 4
    assert payload["splits"]["train"]["question_types"] == {
        "choice": 2,
        "noul": 1,
        "score": 1,
    }
    assert payload["splits"]["train"]["prompt_token_lengths"] == {
        "count": 3,
        "min": 3,
        "max": 8,
        "mean": 16 / 3,
        "p50": 5,
        "p95": 8,
    }
    assert payload["splits"]["train"]["sources"]["nimble"]["rows"] == 2
    assert payload["splits"]["train"]["sources"]["nimble"]["logical_token_lengths"] == {
        "count": 2,
        "min": 7,
        "max": 12,
        "mean": 9.5,
        "p50": 7,
        "p95": 12,
    }
    assert payload["splits"]["eval"]["question_types"] == {"choice": 2}


def test_audit_uses_output_dir_when_prepared_path_is_absent(tmp_path):
    cfg = _cfg(tmp_path)
    del cfg["dataset_prepared_path"]

    audit = build_preparation_audit(cfg, (), (), {})
    path = write_preparation_audit(cfg, (), (), {})

    assert path == tmp_path / "output" / PREPARATION_AUDIT_FILENAME
    assert audit["splits"]["train"]["prompt_token_lengths"]["count"] == 0
    assert audit["splits"]["eval"]["logical_token_lengths"]["p95"] is None


def test_audit_path_can_be_run_local_while_prepared_cache_is_shared(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["_decision_preparation_audit_dir"] = str(tmp_path / "run" / "prepared")

    assert preparation_audit_path(cfg) == tmp_path / "run" / "prepared" / (
        PREPARATION_AUDIT_FILENAME
    )
