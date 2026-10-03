"""Evaluation-only decision loading preserves declared heldout splits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from axolotl.integrations.decision import (
    data_audit,
    datasets,
    prepared_cache,
)
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)


class Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 11
    unk_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt and not enable_thinking
        assert len(messages) == 2
        return [99]


def _spec() -> DiffusionSpec:
    return DiffusionSpec(
        noise=DiffusionNoise.UNIFORM,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=False,
        max_canvas=128,
        max_context=1024,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=MaskTokenPolicy.NONE,
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        reduction_scope=ReductionScope.MICROBATCH,
    )


def _record(source: str, identifier: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "source": source,
        "group": f"{source}-{identifier}",
        "state": f"state-{identifier}",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "options": ["one", "two"],
            }
        },
        "labels": {"q": {"kind": "hard", "gold_idx": 0}},
    }


def _cfg(tmp_path) -> dict[str, Any]:
    return {
        "seed": 23,
        "micro_batch_size": 2,
        "dataset_prepared_path": str(tmp_path / "prepared"),
        "model_config": {"vocab_size": 256, "mask_token_id": 9},
        "diffusion": {"canvas_width": 128, "mask_token_id": 9},
        "decision": {
            "max_questions_per_canvas": 20,
            "labels": {},
            "mixture": {"per_batch_stratified": True},
        },
        "datasets": [
            {"path": "train", "type": "decision.nimble", "split": "train"},
            {
                "path": "calibration",
                "type": "decision.nimble",
                "split": "calibration",
            },
        ],
        "test_datasets": [
            {"path": "dev", "type": "decision.nimble", "split": "dev"},
            {"path": "test", "type": "decision.nimble", "split": "test"},
            {"path": "ood", "type": "decision.nimble", "split": "ood"},
        ],
    }


def _patch_sources(monkeypatch, rows_by_path):
    calls = []

    def rows(entry):
        path = entry["path"]
        calls.append(path)
        return rows_by_path[path]

    monkeypatch.setattr(datasets, "_rows", rows)
    monkeypatch.setattr(
        datasets, "normalize_record", lambda _adapter, row, **_kwargs: dict(row)
    )
    monkeypatch.setattr(datasets, "get_model_support_for_cfg", lambda _cfg: object())
    monkeypatch.setattr(
        datasets,
        "resolve_model_support",
        lambda _support: SimpleNamespace(diffusion=_spec()),
    )
    return calls


def test_evaluation_dev_canvas_matches_existing_explicit_dev_path(
    tmp_path, monkeypatch
):
    rows = {
        "train": [_record("train", "train")],
        "calibration": [_record("calibration", "calibration")],
        "dev": [_record("dev", "dev")],
        "test": [_record("test", "test")],
        "ood": [_record("ood", "ood")],
    }
    calls = _patch_sources(monkeypatch, rows)
    cfg = _cfg(tmp_path)

    class CacheTokenizer(Tokenizer):
        def __len__(self):
            return 256

    tokenizer = CacheTokenizer()

    existing = datasets.load_decision_datasets(cfg, tokenizer=tokenizer).eval_dataset
    assert existing is not None
    calls.clear()
    selected = datasets.load_decision_evaluation_dataset(cfg, tokenizer, "dev")

    assert calls == ["dev"]
    assert [row["record"]["id"] for row in selected] == ["dev"]
    assert [row["canvas"] for row in selected] == [row["canvas"] for row in existing]
    assert selected.manifest["selected_split"] == "dev"
    assert selected.manifest["selected_source_inputs"] == [
        {
            "collection": "test_datasets",
            "index": 0,
            "adapter": "decision.nimble",
            "path": "dev",
            "revision": None,
            "name": None,
            "split": "dev",
        }
    ]
    audit = json.loads(
        (
            tmp_path / "prepared" / "decision_dev_preparation_audit.json"
        ).read_text()
    )
    assert audit["source_inputs"] == selected.manifest["selected_source_inputs"]
    assert audit["preparation_manifest"]["selected_split"] == "dev"


@pytest.mark.parametrize(
    ("split", "path", "identifier"),
    [
        ("test", "test", "test"),
        ("calibration", "calibration", "calibration"),
        ("ood", "ood", "ood"),
    ],
)
def test_evaluation_loader_selects_only_the_requested_heldout_split(
    tmp_path, monkeypatch, split, path, identifier
):
    rows = {
        "train": [_record("train", "train")],
        "calibration": [_record("calibration", "calibration")],
        "dev": [_record("dev", "dev")],
        "test": [_record("test", "test")],
        "ood": [_record("ood", "ood")],
    }
    calls = _patch_sources(monkeypatch, rows)

    selected = datasets.load_decision_evaluation_dataset(
        _cfg(tmp_path), Tokenizer(), split
    )

    assert calls == [path]
    assert [row["record"]["id"] for row in selected] == [identifier]
    assert selected.manifest["input_rows"] == 1
    assert selected.manifest["grouped_rows"] == 1
    assert selected.manifest["eval_rows"] == 1
    assert selected.manifest["budget_drops"] == {"eval": {"logical": 0, "physical": 0}}


def test_evaluation_loader_rejects_undeclared_or_train_splits(tmp_path, monkeypatch):
    rows = {
        "train": [_record("train", "train")],
        "calibration": [_record("calibration", "calibration")],
        "dev": [_record("dev", "dev")],
        "test": [_record("test", "test")],
        "ood": [_record("ood", "ood")],
    }
    _patch_sources(monkeypatch, rows)

    with pytest.raises(ValueError, match="split must be one of"):
        datasets.load_decision_evaluation_dataset(_cfg(tmp_path), Tokenizer(), "train")
    with pytest.raises(ValueError, match="no explicitly declared"):
        datasets.load_decision_evaluation_dataset(
            _cfg(tmp_path), Tokenizer(), "validation"
        )


def test_explicit_evaluation_reuses_only_selected_split_cache(tmp_path, monkeypatch):
    source = tmp_path / "test.jsonl"
    source.write_text("{}\n")
    rows = {
        "train": [_record("train", "train")],
        "json": [_record("test", "test")],
    }
    calls = _patch_sources(monkeypatch, rows)
    cfg = _cfg(tmp_path)
    cfg["datasets"] = [
        {"path": "train", "type": "decision.nimble", "split": "train"}
    ]
    cfg["test_datasets"] = [
        {
            "path": "json",
            "data_files": str(source),
            "type": "decision.nimble",
            "split": "test",
        }
    ]
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("tokenizer")

    class CacheTokenizer(Tokenizer):
        def __len__(self):
            return 256

    tokenizer = CacheTokenizer()
    tokenizer.name_or_path = str(root)
    tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "backend")
    tokenizer.get_vocab = lambda: {"token": 0}
    tokenizer.get_added_vocab = lambda: {}

    first = datasets.load_decision_evaluation_dataset(cfg, tokenizer, "test")
    assert calls == ["json"]
    cold_audit = json.loads(
        (
            tmp_path / "prepared" / "decision_test_preparation_audit.json"
        ).read_text()
    )
    cfg["_decision_preparation_audit_dir"] = str(tmp_path / "run-two" / "prepared")
    calls.clear()
    audit_builds = {"cache_rows": 0, "sidecar": 0}
    cache_build = prepared_cache.build_preparation_audit
    sidecar_build = data_audit.build_preparation_audit

    def count_cache_build(_cfg, train_rows, eval_rows, manifest, **kwargs):
        if train_rows or eval_rows:
            audit_builds["cache_rows"] += 1
        return cache_build(_cfg, train_rows, eval_rows, manifest, **kwargs)

    def count_sidecar_build(*args, **kwargs):
        audit_builds["sidecar"] += 1
        return sidecar_build(*args, **kwargs)

    monkeypatch.setattr(prepared_cache, "build_preparation_audit", count_cache_build)
    monkeypatch.setattr(data_audit, "build_preparation_audit", count_sidecar_build)
    second = datasets.load_decision_evaluation_dataset(cfg, tokenizer, "test")
    assert not calls
    assert audit_builds == {"cache_rows": 1, "sidecar": 0}
    assert [row["canvas"] for row in second] == [row["canvas"] for row in first]
    assert second.manifest["preparation_audit"].startswith(
        str(tmp_path / "run-two" / "prepared")
    )
    warm_audit = json.loads(Path(second.manifest["preparation_audit"]).read_text())
    assert warm_audit == cold_audit

    for mutate in (
        lambda: source.write_text('{"changed": true}\n'),
        lambda: cfg["decision"]["labels"].update(codebook="spreadsheet151"),
        lambda: cfg.update(sequence_len=256),
        lambda: cfg.update(eval_batch_size=3),
        lambda: cfg.update(batch_flattening=True),
    ):
        mutate()
        calls.clear()
        datasets.load_decision_evaluation_dataset(cfg, tokenizer, "test")
        assert calls == ["json"]
