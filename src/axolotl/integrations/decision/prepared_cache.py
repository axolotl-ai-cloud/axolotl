"""Versioned local prepared-row cache for decision training."""

from __future__ import annotations

import hashlib
import importlib.util
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
from filelock import FileLock
from huggingface_hub import hf_hub_download

from axolotl.utils.data.shared import (
    get_prepared_dataset_path,
    load_preprocessed_dataset,
    save_preprocessed_dataset,
)
from axolotl.utils.dict import DictDefault

from .data_audit import build_preparation_audit
from .row_codec import (
    _row_from_json as _row_from_json,
    _row_to_json as _row_to_json,
    dataset_to_rows,
    rows_to_dataset,
)

SCHEMA_VERSION = 3
PRODUCER = "decision-prepared-arrow-v1"
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


def _image_identity(source_paths: Sequence[Path]) -> list[tuple[str, str]] | None:
    """Hash local image bytes referenced by JSONL sources; remote media is uncached."""
    images: list[Path] = []
    for source in source_paths:
        try:
            handle = source.open(encoding="utf-8")
        except OSError:
            return None
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return None
                values = row.get("images") if isinstance(row, Mapping) else None
                if values is None:
                    continue
                if not isinstance(values, list):
                    return None
                for image in values:
                    if not isinstance(image, str) or not image:
                        return None
                    if "://" in image:
                        return None
                    path = Path(image)
                    # Processor paths are interpreted from the preparation cwd.
                    path = path.resolve()
                    if not path.is_file():
                        return None
                    images.append(path.resolve())
    return [(str(path), _file_hash(path)) for path in dict.fromkeys(images)]


def _processor_identity(cfg: Any) -> dict[str, Any]:
    spec = importlib.util.find_spec(
        "axolotl.model_support.nemotron_diffusion.processing"
    )
    origin = None if spec is None else spec.origin
    model_source = _value(cfg, "base_model")
    source = Path(model_source) if isinstance(model_source, str) else None
    return {
        "model_source": model_source,
        "settings": _value(cfg, "processor_kwargs"),
        "default_max_image_size": os.environ.get("DEFAULT_MAX_IMAGE_SIZE"),
        "model_revision": _value(_value(cfg, "model_config"), "_commit_hash")
        or _value(cfg, "revision_of_model")
        or _value(cfg, "_commit_hash"),
        "local_processor_files": {
            name: _file_hash(source / name)
            for name in ("image_processing.py", "config.json")
            if source is not None and (source / name).is_file()
        },
        "module_sha256": _file_hash(Path(origin))
        if isinstance(origin, str) and Path(origin).is_file()
        else None,
    }


def _uses_vlm(cfg: Any) -> bool:
    return (
        _value(_value(cfg, "model_config"), "model_type")
        == "nemotron_labs_diffusion_vlm"
    )


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
    image_inputs: list[tuple[str, str]] = []
    if _uses_vlm(cfg):
        image_inputs = _image_identity(source_paths)
        if image_inputs is None:
            return None
    try:
        spec_value = asdict(spec) if is_dataclass(spec) else vars(spec)  # type: ignore[arg-type]
        audit_config = build_preparation_audit(cfg, (), (), {})["config"]
    except (AttributeError, TypeError, ValueError):
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
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
        "processor": _processor_identity(cfg) if _uses_vlm(cfg) else None,
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
            "vocab_sha256": _sha(sorted(vocabulary().items())),
            "added_vocab": sorted(
                getattr(tokenizer, "get_added_vocab", lambda: {})().items()
            ),
            "backend_sha256": hashlib.sha256(backend().encode()).hexdigest(),
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": transformers.__version__,
        },
        "sources": [(str(path.resolve()), _file_hash(path)) for path in source_paths],
        "images": image_inputs,
    }
    if hub_identity is not None:
        payload["tokenizer"]["hub"] = hub_identity
    return _sha(payload), payload


def _cache_cfg(root: Path, cfg: Any | None = None) -> DictDefault:
    return DictDefault(
        dataset_prepared_path=str(root / "decision_cache"),
        dataset_num_proc=1,
        num_dataset_shards_to_save=None,
        push_dataset_to_hub=False,
        skip_prepare_dataset=bool(_value(cfg, "skip_prepare_dataset", False)),
        is_preprocess=bool(_value(cfg, "is_preprocess", False)),
    )


@overload
def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    *,
    include_audit: Literal[True],
    cfg: Any | None = None,
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
    cfg: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None: ...


def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    *,
    include_audit: bool = False,
    cfg: Any | None = None,
) -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
    | tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]
    | None
):
    parent = root / "decision_cache"
    path = parent / f"{key}.json"
    if not path.is_file():
        return None
    cache_cfg = _cache_cfg(root, cfg)
    if cache_cfg.skip_prepare_dataset or cache_cfg.is_preprocess:
        return None
    try:
        with FileLock(str(parent / f"{key}.lock")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["content"]
            if _sha(metadata) != payload["sha256"]:
                return None
            if _sha(metadata["identity"]) != _sha(identity_payload):
                LOG.info("Decision prepared-row cache identity mismatch")
                return None
            if not get_prepared_dataset_path(cache_cfg, key).is_dir():
                return None
            dataset = load_preprocessed_dataset(cache_cfg, key)
            if dataset is None:
                return None
            train, eval_rows = dataset_to_rows(dataset)
            manifest = dict(metadata["manifest"])
            if "stratified_epoch_batches" in manifest:
                manifest["stratified_epoch_batches"] = tuple(
                    tuple(batch) for batch in manifest["stratified_epoch_batches"]
                )
            audit = build_preparation_audit(
                metadata["config"], train, eval_rows, manifest
            )
            if audit["splits"] != metadata["audit"]["splits"]:
                LOG.info("Decision prepared-row cache audit mismatch")
                return None
            if include_audit:
                return train, eval_rows, manifest, dict(metadata["audit"])
            return train, eval_rows, manifest
    except Exception:
        LOG.warning("Decision prepared-row cache could not be read", exc_info=True)
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
    parent = root / "decision_cache"
    parent.mkdir(parents=True, exist_ok=True)
    cache_cfg = _cache_cfg(root, cfg)
    content = {
        "identity": identity_payload,
        "config": audit["config"],
        "manifest": manifest,
        "audit": audit,
        "train_count": len(train),
        "eval_count": len(eval_rows),
    }
    with FileLock(str(parent / f"{key}.lock")):
        dataset = rows_to_dataset(train, eval_rows)
        save_preprocessed_dataset(cache_cfg, dataset, key, "train")
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
            temporary.replace(parent / f"{key}.json")
        finally:
            temporary.unlink(missing_ok=True)
