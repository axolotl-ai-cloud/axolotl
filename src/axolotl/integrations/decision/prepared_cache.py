"""Versioned local prepared-row cache for decision training."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, overload

import tokenizers
import transformers
from huggingface_hub import hf_hub_download

from .data_audit import build_preparation_audit
from .loss import decision_example_from_canvas
from .records import DecisionCanvas, OrdinalMetadata

SCHEMA_VERSION = 1
PRODUCER = "decision-prepared-rows-v2"
LOG = logging.getLogger(__name__)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(item: Any) -> Any:
    if isinstance(item, Enum):
        return item.value
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    if isinstance(item, Path):
        return str(item)
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return str(item)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, Mapping)
        else getattr(value, name, default)
    )


def _entry_identity(
    cfg: Any, selected: Sequence[tuple[str, int, Any]] | None = None
) -> list[dict[str, Any]]:
    entries = []
    candidates = (
        selected
        if selected is not None
        else (
            (collection, index, entry)
            for collection in ("datasets", "test_datasets")
            for index, entry in enumerate(_value(cfg, collection, ()) or ())
        )
    )
    for collection, index, entry in candidates:
        adapter = _value(entry, "type")
        if isinstance(adapter, str) and adapter.startswith("decision."):
            entries.append(
                {
                    "collection": collection,
                    "index": index,
                    "type": adapter,
                    "path": _value(entry, "path"),
                    "split": _value(entry, "split", "train"),
                    "name": _value(entry, "name"),
                    "revision": _value(entry, "revision"),
                    "data_files": _value(entry, "data_files"),
                }
            )
    return entries


def _resolved_tokenizer_revision(cfg: Any, tokenizer: Any) -> str | None:
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    model_config = _value(cfg, "model_config")
    for value in (
        getattr(tokenizer, "_commit_hash", None),
        _value(init_kwargs, "_commit_hash"),
        _value(model_config, "_commit_hash"),
    ):
        if isinstance(value, str) and value:
            return value
    requested_revision = _value(cfg, "revision_of_model")
    if isinstance(requested_revision, str) and re.fullmatch(
        r"[0-9a-fA-F]{40}", requested_revision
    ):
        return requested_revision.lower()
    return None


def _tokenizer_root(
    cfg: Any, tokenizer: Any
) -> tuple[Path, dict[str, str] | None] | None:
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if not isinstance(name_or_path, str):
        return None
    local_root = Path(name_or_path)
    if local_root.is_dir():
        return local_root, None
    revision = _resolved_tokenizer_revision(cfg, tokenizer)
    if revision is None:
        return None
    for filename in (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        try:
            tokenizer_file = Path(
                hf_hub_download(
                    name_or_path,
                    filename=filename,
                    repo_type="model",
                    revision=revision,
                    local_files_only=True,
                )
            )
        except (OSError, ValueError):
            continue
        if tokenizer_file.is_file():
            return tokenizer_file.parent, {
                "repo_id": name_or_path,
                "resolved_revision": revision,
            }
    return None


def identity(
    cfg: Any,
    tokenizer: Any,
    spec: Any,
    source_paths: Sequence[Path],
    *,
    selected_entries: Sequence[tuple[str, int, Any]] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    backend = getattr(getattr(tokenizer, "backend_tokenizer", None), "to_str", None)
    vocabulary = getattr(tokenizer, "get_vocab", None)
    resolved_root = _tokenizer_root(cfg, tokenizer)
    if resolved_root is None or not (callable(backend) and callable(vocabulary)):
        return None
    token_root, hub_identity = resolved_root
    tokenizer_files = sorted(
        path
        for path in token_root.iterdir()
        if path.is_file()
        and path.name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
        }
    )
    if not tokenizer_files or not source_paths:
        return None
    try:
        spec_value = asdict(spec) if is_dataclass(spec) else vars(spec)  # type: ignore[arg-type]
        audit_config = build_preparation_audit(cfg, (), (), {})["config"]
    except (AttributeError, TypeError, ValueError):
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "producer_sha256": _file_hash(Path(__file__)),
        "producer_dependencies": [
            (str(path.relative_to(Path(__file__).parent)), _file_hash(path))
            for path in sorted(Path(__file__).parent.rglob("*.py"))
            if "__pycache__" not in path.parts
        ],
        "config": audit_config,
        "entries": _entry_identity(cfg, selected_entries),
        "scope": dict(scope or {}),
        "preparation": {
            "model_config": _value(cfg, "model_config"),
            "sample_packing": _value(cfg, "sample_packing"),
            "eval_sample_packing": _value(cfg, "eval_sample_packing"),
            "batch_flattening": _value(cfg, "batch_flattening"),
            "eval_batch_size": _value(cfg, "eval_batch_size"),
        },
        "spec": spec_value,
        "tokenizer": {
            "files": [
                (str(path.relative_to(token_root)), _file_hash(path))
                for path in tokenizer_files
            ],
            "special_ids": {
                name: getattr(tokenizer, name, None)
                for name in (
                    "pad_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "unk_token_id",
                    "mask_token_id",
                )
            },
            "vocab_size": len(tokenizer),
            "chat_template": getattr(tokenizer, "chat_template", None),
            "vocab": sorted(vocabulary().items()),
            "added_vocab": sorted(
                getattr(tokenizer, "get_added_vocab", lambda: {})().items()
            ),
            "backend": backend(),
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": transformers.__version__,
        },
        "sources": [(str(path.resolve()), _file_hash(path)) for path in source_paths],
    }
    if hub_identity is not None:
        payload["tokenizer"]["hub"] = hub_identity
    return _sha(payload), payload


def _canvas_to_json(canvas: DecisionCanvas) -> dict[str, Any]:
    data = asdict(canvas)
    return data


def _canvas_from_json(data: Mapping[str, Any]) -> DecisionCanvas:
    ordinal = tuple(
        None
        if item is None
        else OrdinalMetadata(
            levels=tuple(item["levels"]),
            source_ids=tuple(item["source_ids"]),
            candidate_ranks=tuple(item["candidate_ranks"]),
        )
        for item in data.get("ordinal_metadata", ())
    )
    return DecisionCanvas(
        prompt_ids=tuple(data["prompt_ids"]),
        canvas_ids=tuple(data["canvas_ids"]),
        label_positions=tuple(data["label_positions"]),
        allowed_ids=tuple(tuple(value) for value in data["allowed_ids"]),
        question_ids=tuple(data["question_ids"]),
        targets=tuple(data["targets"]),
        pinned_mask=tuple(data["pinned_mask"]),
        semantic_mask=tuple(data["semantic_mask"]),
        slot_mask=tuple(data["slot_mask"]),
        template_length=int(data["template_length"]),
        prompt_slot_mask=tuple(data.get("prompt_slot_mask", ())),
        ordinal_metadata=ordinal,
    )


def _row_to_json(row: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: item
        for key, item in row.items()
        if key not in {"canvas", "decision_example", "slot_plan"}
    }
    value["canvas"] = _canvas_to_json(row["canvas"])
    example = row.get("decision_example")
    if example is not None:
        value["_decision_example_source_weight"] = example.source_weight
    return value


def _row_from_json(data: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(data)
    if isinstance(row.get("record"), dict) and "grouped_record_ids" in row["record"]:
        row["record"] = dict(row["record"])
        row["record"]["grouped_record_ids"] = tuple(row["record"]["grouped_record_ids"])
    row["canvas"] = _canvas_from_json(row["canvas"])
    source_weight = float(
        row.pop("_decision_example_source_weight", row.get("source_weight", 1.0))
    )
    row["decision_example"] = decision_example_from_canvas(
        row["canvas"], source_weight=source_weight
    )
    return row


@overload
def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    *,
    include_audit: Literal[True],
) -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]
    | None
): ...


@overload
def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    *,
    include_audit: Literal[False] = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None: ...


def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    *,
    include_audit: bool = False,
) -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
    | tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]
    | None
):
    path = root / "decision_cache" / f"{key}.json"
    try:
        payload = json.loads(path.read_text())
        if _sha(payload["content"]) != payload["sha256"]:
            return None
        metadata = payload["content"]
        if _sha(metadata["identity"]) != _sha(identity_payload):
            LOG.info("Decision prepared-row cache identity mismatch")
            return None
        rows = metadata["rows"]
        train = [_row_from_json(row) for row in rows["train"]]
        eval_rows = [_row_from_json(row) for row in rows["eval"]]
        manifest = dict(metadata["manifest"])
        if "stratified_epoch_batches" in manifest:
            manifest["stratified_epoch_batches"] = tuple(
                tuple(batch) for batch in manifest["stratified_epoch_batches"]
            )
        audit = build_preparation_audit(metadata["config"], train, eval_rows, manifest)
        if (
            audit["splits"] != metadata["audit"]["splits"]
            or _sha(rows) != metadata["rows_sha256"]
        ):
            LOG.info("Decision prepared-row cache payload validation mismatch")
            return None
        if include_audit:
            return train, eval_rows, manifest, dict(metadata["audit"])
        return train, eval_rows, manifest
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def store(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    cfg: Any,
    train: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    audit: Mapping[str, Any] | None = None,
) -> None:
    audit = (
        dict(audit)
        if audit is not None
        else build_preparation_audit(cfg, train, eval_rows, manifest)
    )
    rows = {
        "train": [_row_to_json(row) for row in train],
        "eval": [_row_to_json(row) for row in eval_rows],
    }
    content = {
        "identity": identity_payload,
        "config": audit["config"],
        "manifest": manifest,
        "audit": audit,
        "rows_sha256": _sha(rows),
        "rows": rows,
    }
    parent = root / "decision_cache"
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".tmp", dir=parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"sha256": _sha(content), "content": content},
                handle,
                sort_keys=True,
                default=_json_default,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, parent / f"{key}.json")
    finally:
        if temporary.exists():
            temporary.unlink()
