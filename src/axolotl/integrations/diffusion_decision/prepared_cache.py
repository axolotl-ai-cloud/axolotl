"""Versioned local prepared-row cache for decision training."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import tokenizers
import transformers
from huggingface_hub import hf_hub_download

from ._util import (
    _value,
    atomic_write_text,
    canonical_sha256,
    model_config_overrides,
    sha256_file,
)
from .loss import decision_example_from_canvas
from .records import DecisionCanvas, OrdinalMetadata
from .slots import SlotPlan

SCHEMA_VERSION = 1
PRODUCER = "decision-prepared-rows-v2"
LOG = logging.getLogger(__name__)


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
    return canonical_sha256(value, default=_json_default)


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
        if isinstance(adapter, str) and adapter.startswith("diffusion_decision."):
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
    model_config = model_config_overrides(cfg)
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
        semantic_config = _semantic_config(cfg)
    except (AttributeError, TypeError, ValueError):
        return None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "producer_sha256": sha256_file(Path(__file__)),
        "producer_dependencies": [
            (str(path.relative_to(Path(__file__).parent)), sha256_file(path))
            for path in sorted(Path(__file__).parent.rglob("*.py"))
            if "__pycache__" not in path.parts
        ],
        "config": semantic_config,
        "entries": _entry_identity(cfg, selected_entries),
        "scope": dict(scope or {}),
        "preparation": {
            "model_config": model_config_overrides(cfg),
            "sample_packing": _value(cfg, "sample_packing"),
            "eval_sample_packing": _value(cfg, "eval_sample_packing"),
            "batch_flattening": _value(cfg, "batch_flattening"),
            "eval_batch_size": _value(cfg, "eval_batch_size"),
        },
        "spec": spec_value,
        "tokenizer": {
            "files": [
                (str(path.relative_to(token_root)), sha256_file(path))
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
        "sources": [(str(path.resolve()), sha256_file(path)) for path in source_paths],
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
    plan = row.get("slot_plan")
    if plan is not None:
        value["slot_plan"] = asdict(plan)
    return value


def _row_from_json(data: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(data)
    if isinstance(row.get("record"), dict) and "grouped_record_ids" in row["record"]:
        row["record"] = dict(row["record"])
        row["record"]["grouped_record_ids"] = tuple(row["record"]["grouped_record_ids"])
    row["canvas"] = _canvas_from_json(row["canvas"])
    if "slot_plan" in row:
        plan = row["slot_plan"]
        row["slot_plan"] = SlotPlan(
            ids=tuple(plan["ids"]),
            placement=plan["placement"],
            pinned_mask=tuple(plan["pinned_mask"]),
            update_mask=tuple(plan["update_mask"]),
            loss_mask=tuple(plan["loss_mask"]),
            trainable_token_ids=tuple(plan["trainable_token_ids"]),
        )
    source_weight = float(
        row.pop("_decision_example_source_weight", row.get("source_weight", 1.0))
    )
    row["decision_example"] = decision_example_from_canvas(
        row["canvas"], source_weight=source_weight
    )
    return row


def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    path = root / "diffusion_decision_cache" / f"{key}.json"
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
        if _sha(rows) != metadata["rows_sha256"]:
            LOG.info("Decision prepared-row cache payload validation mismatch")
            return None
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
) -> None:
    rows = {
        "train": [_row_to_json(row) for row in train],
        "eval": [_row_to_json(row) for row in eval_rows],
    }
    content = {
        "identity": identity_payload,
        "config": _semantic_config(cfg),
        "manifest": manifest,
        "rows_sha256": _sha(rows),
        "rows": rows,
    }
    parent = root / "diffusion_decision_cache"
    parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        parent / f"{key}.json",
        json.dumps(
            {"sha256": _sha(content), "content": content},
            sort_keys=True,
            default=_json_default,
        ),
    )
