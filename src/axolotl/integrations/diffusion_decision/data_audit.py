"""Compact, reproducible summaries of prepared decision datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from ._util import _value, atomic_write_text, canonical_json, canonical_sha256

PREPARATION_AUDIT_FILENAME = "diffusion_decision_preparation_audit.json"
PREPARATION_AUDIT_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    raise TypeError(f"cannot serialize decision preparation value {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return canonical_json(_json_value(value))


def _sha256(value: Any) -> str:
    return canonical_sha256(_json_value(value))


def _json_safe_prepared_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if type(value) is dict:
        return all(
            isinstance(key, str) and _json_safe_prepared_value(item)
            for key, item in value.items()
        )
    return type(value) in {list, tuple} and all(
        _json_safe_prepared_value(item) for item in value
    )


def _canonical_prepared_value(value: Any) -> str:
    if _json_safe_prepared_value(value):
        return canonical_json(value)
    return _canonical_json(value)


def _sha256_canonical_items(items: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, item in enumerate(items):
        if index:
            digest.update(b",")
        digest.update(item.encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def _length_summary(lengths: Sequence[int]) -> dict[str, int | float | None]:
    if not lengths:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    ordered = sorted(lengths)

    def percentile(percent: float) -> int:
        return ordered[round((len(ordered) - 1) * percent)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def _question_types(row: Mapping[str, Any]) -> Counter[str]:
    record = row.get("record")
    if not isinstance(record, Mapping):
        return Counter()
    questions = record.get("questions")
    values: Iterable[Any]
    if isinstance(questions, Mapping):
        values = questions.values()
    elif isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)):
        values = questions
    else:
        return Counter()
    result: Counter[str] = Counter()
    for question in values:
        if not isinstance(question, Mapping):
            result["unknown"] += 1
            continue
        kind = question.get("type")
        result[str(kind) if isinstance(kind, str) and kind else "unknown"] += 1
    return result


def _row_lengths(row: Mapping[str, Any]) -> tuple[int, int]:
    canvas = row.get("canvas")
    if canvas is None:
        raise ValueError(
            "decision preparation audit requires each row to include canvas"
        )
    prompt_ids = getattr(canvas, "prompt_ids", None)
    canvas_ids = getattr(canvas, "canvas_ids", None)
    if not isinstance(prompt_ids, Sequence) or not isinstance(canvas_ids, Sequence):
        raise ValueError(
            "decision preparation audit canvas must expose prompt_ids and canvas_ids"
        )
    prompt_length = len(prompt_ids)
    logical_length = row.get("length", prompt_length + len(canvas_ids))
    if isinstance(logical_length, bool) or not isinstance(logical_length, int):
        raise ValueError("decision preparation audit row length must be an integer")
    if logical_length < prompt_length:
        raise ValueError(
            "decision preparation audit logical length cannot precede prompt"
        )
    return prompt_length, logical_length


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(
                "decision preparation audit rows require a nonempty source"
            )
        by_source[source].append(row)

    digest_cache: dict[int, dict[str, str]] = {}

    def summarize(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        prompt_lengths: list[int] = []
        logical_lengths: list[int] = []
        types: Counter[str] = Counter()
        for row in source_rows:
            prompt_length, logical_length = _row_lengths(row)
            prompt_lengths.append(prompt_length)
            logical_lengths.append(logical_length)
            types.update(_question_types(row))
        content = [_canonical_json(row["record"]) for row in source_rows]
        identities = [
            (row["source"], row["record"].get("id"))
            for row in source_rows
            if isinstance(row["record"].get("id"), str) and row["record"]["id"]
        ]
        identity_content: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row, canonical in zip(source_rows, content, strict=True):
            record_id = row["record"].get("id")
            if isinstance(record_id, str) and record_id:
                identity_content[(row["source"], record_id)].add(canonical)
        result = {
            "rows": len(source_rows),
            "scheduled_draws": len(source_rows),
            "unique_normalized_record_contents": len(set(content)),
            "unique_source_record_ids": len(set(identities)),
            "missing_source_record_ids": len(source_rows) - len(identities),
            "source_record_id_content_collisions": sum(
                len(values) > 1 for values in identity_content.values()
            ),
            "questions": sum(types.values()),
            "question_types": dict(sorted(types.items())),
            "prompt_token_lengths": _length_summary(prompt_lengths),
            "logical_token_lengths": _length_summary(logical_lengths),
        }
        result.update(_row_digests(source_rows, digest_cache))
        return result

    sources = {
        source: summarize(source_rows) for source, source_rows in by_source.items()
    }
    aggregate = summarize(rows)
    aggregate["sources"] = dict(sorted(sources.items()))
    return aggregate


def _row_digests(
    rows: Sequence[Mapping[str, Any]], cache: dict[int, dict[str, str]]
) -> dict[str, Any]:
    values = [_prepared_row_digest_values(row, cache) for row in rows]
    return {
        "row_digest_schema_version": 1,
        "row_digest_algorithm": "sha256-canonical-json-v1",
        "records_sha256": _sha256_canonical_items(
            [value["record"] for value in values]
        ),
        "prompts_sha256": _sha256_canonical_items(
            [value["prompt"] for value in values]
        ),
        "questions_candidates_targets_sha256": _sha256_canonical_items(
            [value["question_candidates_targets"] for value in values]
        ),
        "semantic_record_prompt_candidate_sha256": _sha256_canonical_items(
            [value["semantic"] for value in values]
        ),
        "full_canvas_sha256": _sha256_canonical_items(
            [value["layout"] for value in values]
        ),
    }


def _prepared_row_digest_values(
    row: Mapping[str, Any], cache: dict[int, dict[str, str]]
) -> dict[str, str]:
    key = id(row)
    if key in cache:
        return cache[key]
    canvas = row["canvas"]
    record = row["record"]
    question_candidates_targets = {
        "question_ids": canvas.question_ids,
        "allowed_ids": canvas.allowed_ids,
        "targets": canvas.targets,
    }
    layout = {
        "prompt_ids": canvas.prompt_ids,
        "canvas_ids": canvas.canvas_ids,
        "label_positions": canvas.label_positions,
        "allowed_ids": canvas.allowed_ids,
        "question_ids": canvas.question_ids,
        "targets": canvas.targets,
        "pinned_mask": canvas.pinned_mask,
        "semantic_mask": canvas.semantic_mask,
        "slot_mask": canvas.slot_mask,
        "template_length": canvas.template_length,
        "prompt_slot_mask": canvas.prompt_slot_mask,
        "ordinal_metadata": canvas.ordinal_metadata,
    }
    result = {
        "record": _canonical_prepared_value(record),
        "prompt": _canonical_prepared_value(canvas.prompt_ids),
        "question_candidates_targets": _canonical_prepared_value(
            question_candidates_targets
        ),
        "semantic": _canonical_prepared_value(
            {
                "source": row["source"],
                "record": record,
                "prompt_ids": canvas.prompt_ids,
                "question_candidates_targets": question_candidates_targets,
            }
        ),
        "layout": _canonical_prepared_value(layout),
    }
    cache[key] = result
    return result


def _source_inputs(cfg: Any) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for collection in ("datasets", "test_datasets"):
        entries = _value(cfg, collection, ()) or ()
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"{collection} must be a sequence")
        for index, entry in enumerate(entries):
            adapter = _value(entry, "type")
            if not isinstance(adapter, str) or not adapter.startswith(
                "diffusion_decision."
            ):
                continue
            item = {
                "collection": collection,
                "index": index,
                "adapter": adapter,
                "path": _value(entry, "path"),
                "revision": _value(entry, "revision"),
                "name": _value(entry, "name"),
                "split": _value(entry, "split", "train"),
            }
            data_files = _value(entry, "data_files")
            if data_files is not None:
                item["data_files"] = _json_value(data_files)
            inputs.append(item)
    return inputs


def _semantic_config(cfg: Any) -> dict[str, Any]:
    fields = (
        "base_model",
        "revision_of_model",
        "model_config_type",
        "seed",
        "sequence_len",
        "micro_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "attn_implementation",
        "sample_packing",
        "eval_sample_packing",
        "batch_flattening",
    )
    result = {
        name: _json_value(value)
        for name in fields
        if (value := _value(cfg, name)) is not None
    }
    for name in ("diffusion_lm", "diffusion_decision"):
        value = _value(cfg, name)
        if value is not None:
            result[name] = _json_value(value)
    return result


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    serialized = _json_value(manifest)
    if not isinstance(serialized, dict):
        raise ValueError("decision preparation manifest must be a mapping")
    batches = serialized.pop("stratified_epoch_batches", None)
    if batches is not None:
        if not isinstance(batches, list):
            raise ValueError("stratified_epoch_batches must be a sequence")
        serialized["stratified_epoch_batches_summary"] = {
            "batches": len(batches),
            "examples": sum(len(batch) for batch in batches),
            "sha256": _sha256(batches),
        }
    return serialized


def build_preparation_audit(
    cfg: Any,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    source_inputs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-safe record for one prepared decision dataset."""
    config = _semantic_config(cfg)
    return {
        "schema_version": PREPARATION_AUDIT_SCHEMA_VERSION,
        "config_sha256": _sha256(config),
        "config": config,
        "source_inputs": (
            _json_value(source_inputs)
            if source_inputs is not None
            else _source_inputs(cfg)
        ),
        "preparation_manifest": _manifest_summary(manifest),
        "splits": {
            "train": _split_summary(train_rows),
            "eval": _split_summary(eval_rows),
        },
    }


def preparation_audit_path(
    cfg: Any, filename: str = PREPARATION_AUDIT_FILENAME
) -> Path:
    """Return the configured directory's deterministic preparation-audit path."""
    parent = (
        _value(cfg, "_decision_preparation_audit_dir")
        or _value(cfg, "dataset_prepared_path")
        or _value(cfg, "output_dir")
    )
    if not isinstance(parent, (str, Path)) or not str(parent):
        raise ValueError(
            "decision preparation audit requires dataset_prepared_path or output_dir"
        )
    if not filename or Path(filename).name != filename:
        raise ValueError("decision preparation audit filename must be a basename")
    return Path(parent) / filename


def write_preparation_audit(
    cfg: Any,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    source_inputs: Sequence[Mapping[str, Any]] | None = None,
    filename: str = PREPARATION_AUDIT_FILENAME,
    audit: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically persist one compact preparation audit beside prepared artifacts."""
    path = preparation_audit_path(cfg, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        dict(audit)
        if audit is not None
        else build_preparation_audit(
            cfg, train_rows, eval_rows, manifest, source_inputs=source_inputs
        )
    )
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return path
