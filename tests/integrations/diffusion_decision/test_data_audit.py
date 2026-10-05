"""Tests for compact decision-dataset preparation provenance."""

import json
from collections import UserDict
from copy import deepcopy
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import axolotl.integrations.diffusion_decision.data_audit as data_audit
from axolotl.integrations.diffusion_decision.data_audit import (
    PREPARATION_AUDIT_FILENAME,
    PREPARATION_AUDIT_SCHEMA_VERSION,
    build_preparation_audit,
    preparation_audit_path,
    write_preparation_audit,
)
from axolotl.utils.dict import DictDefault

from tests.integrations.diffusion_decision.helpers import (
    make_canvas,
)


def _canvas(prompt_length, width=4):
    return make_canvas(
        range(prompt_length),
        range(width),
        (1,),
        allowed=(4, 5),
        question_ids=("q1",),
        targets=(0,),
        pinned_mask=(True,) * width,
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
                    "type": "diffusion_decision.jsonl",
                },
                {"path": "ignored", "type": "chat_template"},
            ],
            "test_datasets": [
                {
                    "path": "org/eval-set",
                    "name": "heldout",
                    "revision": "data-r2",
                    "split": "test",
                    "type": "diffusion_decision.procedural",
                }
            ],
        }
    )


def test_audit_persists_compact_manifest_and_per_source_distributions(tmp_path):
    cfg = _cfg(tmp_path)
    manifest = {
        "train_rows": 3,
        "mixture_seed": 42,
        "stratified_epoch_batches": (("alpha", "beta"), ("alpha",)),
    }
    train_rows = [
        _row("alpha", 3, 7, ("choice",)),
        _row("alpha", 8, 12, ("noul", "score")),
        _row("beta", 5, 9, ("choice",)),
    ]
    eval_rows = [_row("beta", 6, 10, ("choice", "choice"))]

    path = write_preparation_audit(cfg, train_rows, eval_rows, manifest)

    assert path == preparation_audit_path(cfg)
    assert path.name == PREPARATION_AUDIT_FILENAME
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PREPARATION_AUDIT_SCHEMA_VERSION
    assert len(payload["config_sha256"]) == 64
    assert "credential_that_must_not_be_persisted" not in payload["config"]
    assert payload["source_inputs"] == [
        {
            "adapter": "diffusion_decision.jsonl",
            "collection": "datasets",
            "data_files": {"train": "/data/train.jsonl"},
            "index": 0,
            "name": None,
            "path": "json",
            "revision": "data-r1",
            "split": "train",
        },
        {
            "adapter": "diffusion_decision.procedural",
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
        "sha256": "d81e39a537766c409b135b81bf6fb5bb437846ee132b2d2c87c5f924bc4ce7ed",
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
    assert payload["splits"]["train"]["sources"]["alpha"]["rows"] == 2
    assert payload["splits"]["train"]["sources"]["alpha"]["logical_token_lengths"] == {
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


def test_semantic_and_layout_digests_distinguish_content_but_not_slot_layout(tmp_path):
    cfg = _cfg(tmp_path)
    first = _row("alpha", 3, 7, ("choice",))
    first["record"]["state"] = "line\u2028separator"
    repeated = deepcopy(first)
    slot_only = deepcopy(first)
    slot_only["canvas"] = replace(
        first["canvas"],
        canvas_ids=(91, *first["canvas"].canvas_ids[1:]),
        label_positions=(2,),
        slot_mask=(True, False, False, False),
    )
    changed = deepcopy(first)
    changed["record"]["state"] = "other"
    changed_question_ids = deepcopy(first)
    changed_question_ids["canvas"] = replace(first["canvas"], question_ids=("renamed",))
    changed_prompt = deepcopy(first)
    changed_prompt["canvas"] = replace(first["canvas"], prompt_ids=(91, 92, 93))
    changed_candidates = deepcopy(first)
    changed_candidates["canvas"] = replace(first["canvas"], allowed_ids=((5, 4),))
    changed_target = deepcopy(first)
    changed_target["canvas"] = replace(first["canvas"], targets=(1,))
    collision = deepcopy(first)
    collision["record"]["state"] = "same-id-different-content"
    missing_id = deepcopy(first)
    del missing_id["record"]["id"]

    baseline = build_preparation_audit(cfg, [first, repeated], (), {})["splits"][
        "train"
    ]
    slots = build_preparation_audit(cfg, [slot_only, repeated], (), {})["splits"][
        "train"
    ]
    mutated = build_preparation_audit(cfg, [changed, repeated], (), {})["splits"][
        "train"
    ]
    reversed_rows = build_preparation_audit(cfg, [repeated, changed], (), {})["splits"][
        "train"
    ]
    ordered_rows = build_preparation_audit(cfg, [changed, repeated], (), {})["splits"][
        "train"
    ]
    changed_layout = build_preparation_audit(
        cfg, [changed_question_ids, repeated], (), {}
    )["splits"]["train"]
    prompt_digest = build_preparation_audit(cfg, [changed_prompt], (), {})["splits"][
        "train"
    ]
    candidate_digest = build_preparation_audit(cfg, [changed_candidates], (), {})[
        "splits"
    ]["train"]
    target_digest = build_preparation_audit(cfg, [changed_target], (), {})["splits"][
        "train"
    ]
    identity_audit = build_preparation_audit(
        cfg, [first, collision, missing_id], (), {}
    )["splits"]["train"]

    assert baseline["scheduled_draws"] == 2
    assert baseline["unique_normalized_record_contents"] == 1
    assert baseline["unique_source_record_ids"] == 1
    assert baseline["source_record_id_content_collisions"] == 0
    assert baseline["row_digest_schema_version"] == 1
    assert baseline["row_digest_algorithm"] == "sha256-canonical-json-v1"
    assert (
        baseline["semantic_record_prompt_candidate_sha256"]
        == slots["semantic_record_prompt_candidate_sha256"]
    )
    assert baseline["full_canvas_sha256"] != slots["full_canvas_sha256"]
    assert baseline["full_canvas_sha256"] != changed_layout["full_canvas_sha256"]
    assert baseline["prompts_sha256"] != prompt_digest["prompts_sha256"]
    assert (
        baseline["questions_candidates_targets_sha256"]
        != candidate_digest["questions_candidates_targets_sha256"]
    )
    assert (
        baseline["questions_candidates_targets_sha256"]
        != target_digest["questions_candidates_targets_sha256"]
    )
    assert baseline["records_sha256"] != mutated["records_sha256"]
    assert (
        baseline["semantic_record_prompt_candidate_sha256"]
        != mutated["semantic_record_prompt_candidate_sha256"]
    )
    assert ordered_rows["records_sha256"] != reversed_rows["records_sha256"]
    assert identity_audit["unique_source_record_ids"] == 1
    assert identity_audit["missing_source_record_ids"] == 1
    assert identity_audit["source_record_id_content_collisions"] == 1


def test_audit_json_roundtrips_u2028_without_losing_digest_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["base_model"] = "native\u2028model"
    path = write_preparation_audit(cfg, [_row("alpha", 3, 7, ("choice",))], (), {})

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["config"]["base_model"] == "native\u2028model"
    assert payload["splits"]["train"]["records_sha256"]


def test_prepared_digest_fast_path_matches_legacy_hashes_and_falls_back_safely(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    rows = [_row("alpha", 3, 7, ("choice",)), _row("beta", 4, 8, ("noul",))]
    audit = build_preparation_audit(cfg, rows, (), {})["splits"]["train"]
    records = [row["record"] for row in rows]
    prompts = [row["canvas"].prompt_ids for row in rows]
    candidates = [
        {
            "question_ids": row["canvas"].question_ids,
            "allowed_ids": row["canvas"].allowed_ids,
            "targets": row["canvas"].targets,
        }
        for row in rows
    ]
    semantic = [
        {
            "source": row["source"],
            "record": record,
            "prompt_ids": prompt,
            "question_candidates_targets": candidate,
        }
        for row, record, prompt, candidate in zip(
            rows, records, prompts, candidates, strict=True
        )
    ]
    layouts = [
        {
            "prompt_ids": row["canvas"].prompt_ids,
            "canvas_ids": row["canvas"].canvas_ids,
            "label_positions": row["canvas"].label_positions,
            "allowed_ids": row["canvas"].allowed_ids,
            "question_ids": row["canvas"].question_ids,
            "targets": row["canvas"].targets,
            "pinned_mask": row["canvas"].pinned_mask,
            "semantic_mask": row["canvas"].semantic_mask,
            "slot_mask": row["canvas"].slot_mask,
            "template_length": row["canvas"].template_length,
            "prompt_slot_mask": row["canvas"].prompt_slot_mask,
            "ordinal_metadata": row["canvas"].ordinal_metadata,
        }
        for row in rows
    ]
    assert audit["records_sha256"] == data_audit._sha256(records)
    assert audit["prompts_sha256"] == data_audit._sha256(prompts)
    assert audit["questions_candidates_targets_sha256"] == data_audit._sha256(
        candidates
    )
    assert audit["semantic_record_prompt_candidate_sha256"] == data_audit._sha256(
        semantic
    )
    assert audit["full_canvas_sha256"] == data_audit._sha256(layouts)

    class Value(Enum):
        ONE = "one"

    for value in (
        {True: "yes", None: "none"},
        {"path": Path("/tmp/x")},
        (Value.ONE,),
        {"nested": MappingProxyType({"value": 1})},
        {"nested": UserDict({"value": 1})},
        {"nested": range(3)},
    ):
        assert data_audit._canonical_prepared_value(
            value
        ) == data_audit._canonical_json(value)
    assert not data_audit._json_safe_prepared_value(
        {"nested": MappingProxyType({"value": 1})}
    )
