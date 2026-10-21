"""Versioned local prepared-row cache for decision training."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
import os
import pickle  # nosec B403 - only our own cache files, read through an allowlist
import re
import struct
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import tokenizers
import transformers
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from axolotl import processing_strategies

from ._util import (
    _value,
    canonical_sha256,
    model_config_overrides,
    sha256_file,
)
from .records import DecisionCanvas

SCHEMA_VERSION = 3
PRODUCER = "decision-canvas-layer-v5"
CACHE_DIR = "diffusion_decision_cache"
CACHE_SUFFIX = ".pkl"
# File layout: MAGIC | rows pickle | header JSON | u64 header length | MAGIC.
# The rows section is one pickle (protocol 5) of {"train", "eval", "manifest"}: the
# canvas layer (per-source pools, or premixed rows), unfiltered eval rows and the
# base manifest. Pools carry each training record's pre-tokenized canvas segments
# (segments.py) so re-permuted draws are assembled without the tokenizer. Batch settings are not part of the key; they only shape the batch
# plan built over the layer at startup (batch_plan.py).
# The trailing header carries the identity hash and the rows section's length and
# sha256, so identity is checked before any rows are deserialized and a truncated
# file is rejected by its size alone.
MAGIC = b"AXDDPC02"
_TRAILER = struct.Struct("<Q8s")
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


# Read only by the batch plan (mixture draws, grouping, budget filter), never by
# canvas preparation, so changing them reuses the canvas layer.
_PLAN_ONLY_MIXTURE_FIELDS = ("weights", "temperature", "per_batch_stratified")


def _semantic_config(cfg: Any) -> dict[str, Any]:
    fields = (
        "base_model",
        "revision_of_model",
        "model_config_type",
        "seed",
        "sequence_len",
        "attn_implementation",
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
    decision = result.get("diffusion_decision")
    if isinstance(decision, dict):
        # How a cache is checked does not change what it holds.
        decision.pop("prepared_cache_verify", None)
        mixture = decision.get("mixture")
        if isinstance(mixture, dict):
            for name in _PLAN_ONLY_MIXTURE_FIELDS:
                mixture.pop(name, None)
    return result


def prepared_cache_verify(cfg: Any) -> bool:
    return bool(
        _value(_value(cfg, "diffusion_decision"), "prepared_cache_verify", False)
    )


def _source_identity(path: Path, verify: bool) -> tuple[Any, ...]:
    resolved = path.resolve()
    if verify:
        return (str(resolved), sha256_file(path))
    stat = resolved.stat()
    return (str(resolved), stat.st_size, stat.st_mtime_ns)


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
    return _cached_snapshot_revision(
        getattr(tokenizer, "name_or_path", None),
        requested_revision if isinstance(requested_revision, str) else None,
    )


def _cached_snapshot_revision(
    name_or_path: Any, requested_revision: str | None = None
) -> str | None:
    """Recover the commit hash from the local Hub cache when the tokenizer carries none.

    transformers 5 tokenizers no longer expose ``_commit_hash``; the snapshot
    directory the tokenizer files were loaded from still names the commit. The
    lookup follows the same ref the tokenizer loader resolved
    (``revision_of_model``, else ``main``), so a branch or tag pin maps to its
    own snapshot rather than ``main``'s.
    """
    if not isinstance(name_or_path, str) or not name_or_path:
        return None
    for filename in ("tokenizer_config.json", "tokenizer.json"):
        try:
            cached = try_to_load_from_cache(
                name_or_path,
                filename=filename,
                revision=requested_revision,
                repo_type="model",
            )
        except (OSError, ValueError):
            continue
        if not isinstance(cached, str):
            continue
        parts = Path(cached).parts
        if "snapshots" in parts:
            revision = parts[parts.index("snapshots") + 1]
            if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                return revision.lower()
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
    image_paths: Sequence[Path] = (),
    verify: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    """Cache key and identity payload, or None when the inputs cannot be pinned.

    Sources are keyed by content sha256 when ``verify`` is set, else by resolved
    path, size and mtime (hashing multi-GB sources dominated cache hits).
    """
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
        ]
        + [
            (
                "axolotl/processing_strategies.py",
                sha256_file(Path(processing_strategies.__file__)),
            )
        ],
        "config": semantic_config,
        "entries": _entry_identity(cfg, selected_entries),
        "scope": dict(scope or {}),
        "preparation": {"model_config": model_config_overrides(cfg)},
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
        "source_identity": "sha256" if verify else "stat",
        "sources": [_source_identity(path, verify) for path in source_paths],
        "images": [(str(path.resolve()), sha256_file(path)) for path in image_paths],
    }
    if hub_identity is not None:
        payload["tokenizer"]["hub"] = hub_identity
    return _sha(payload), payload


def _canvas_to_json(canvas: DecisionCanvas) -> dict[str, Any]:
    return asdict(canvas)


_PACKAGE = __name__.rpartition(".")[0]
_SAFE_PREFIXES = (f"{_PACKAGE}.", "axolotl.integrations.diffusion_lm.")
_SAFE_GLOBALS = {
    "builtins": {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "range",
        "set",
        "slice",
        "str",
        "tuple",
    },
    "collections": {"OrderedDict", "defaultdict", "deque"},
    "pathlib": {"Path", "PosixPath", "PurePosixPath", "PureWindowsPath", "WindowsPath"},
    "numpy": {"dtype", "ndarray"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.numeric": {"_frombuffer"},
    "numpy._core.numeric": {"_frombuffer"},
}


class _CacheUnpickler(pickle.Unpickler):  # nosec B301
    """Resolve only plain containers, numpy arrays and this plugin's own types."""

    def find_class(self, module: str, name: str) -> Any:
        if name in _SAFE_GLOBALS.get(module, ()):
            return super().find_class(module, name)
        if module.startswith(_SAFE_PREFIXES):
            value = super().find_class(module, name)
            if isinstance(value, type):
                return value
        raise pickle.UnpicklingError(
            f"prepared-row cache references disallowed global {module}.{name}"
        )


class _HashingWriter(io.RawIOBase):
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.length = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = memoryview(data)
        self.digest.update(view)
        self._handle.write(view)
        self.length += view.nbytes
        return view.nbytes


def _cache_path(root: Path, key: str) -> Path:
    return root / CACHE_DIR / f"{key}{CACHE_SUFFIX}"


def _read_header(handle: Any, size: int) -> dict[str, Any] | None:
    if size < 2 * len(MAGIC) + _TRAILER.size or handle.read(len(MAGIC)) != MAGIC:
        return None
    handle.seek(size - _TRAILER.size)
    header_length, magic = _TRAILER.unpack(handle.read(_TRAILER.size))
    header_start = size - _TRAILER.size - header_length
    if magic != MAGIC or header_start < len(MAGIC):
        return None
    handle.seek(header_start)
    header = json.loads(handle.read(header_length).decode("utf-8"))
    if not isinstance(header, dict) or header.get("rows_length") != (
        header_start - len(MAGIC)
    ):
        return None
    return header


def _rows_sha256(handle: Any, length: int) -> str:
    digest = hashlib.sha256()
    handle.seek(len(MAGIC))
    remaining = length
    while remaining:
        chunk = handle.read(min(remaining, 1 << 24))
        if not chunk:
            raise EOFError("prepared-row cache rows section is truncated")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def load(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    *,
    verify: bool = False,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]] | None:
    """Return the cached canvas layer (train, eval, manifest), or None on a miss.

    The identity is always compared. ``verify`` additionally hashes the rows
    section against the sha256 recorded at store time; without it a damaged file
    is still rejected when its size, framing or pickle stream does not hold up.
    """
    path = _cache_path(root, key)
    try:
        with path.open("rb", buffering=1 << 24) as handle:
            size = os.fstat(handle.fileno()).st_size
            header = _read_header(handle, size)
            if (
                header is None
                or header.get("schema_version") != SCHEMA_VERSION
                or header.get("producer") != PRODUCER
            ):
                LOG.info("Decision prepared-row cache format mismatch")
                return None
            if header.get("identity_sha256") != _sha(identity_payload):
                LOG.info("Decision prepared-row cache identity mismatch")
                return None
            if verify and _rows_sha256(handle, header["rows_length"]) != header.get(
                "rows_sha256"
            ):
                LOG.info("Decision prepared-row cache payload validation mismatch")
                return None
            handle.seek(len(MAGIC))
            # Millions of long-lived acyclic containers: cyclic GC passes during the
            # load, and the first full pass after it, find nothing. Load with GC off,
            # then move what is tracked to the permanent generation so later passes
            # skip the rows (refcounting still frees them when the dataset goes).
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                content = _CacheUnpickler(handle).load()
                gc.freeze()
            finally:
                if gc_enabled:
                    gc.enable()
            if handle.tell() != len(MAGIC) + header["rows_length"]:
                return None
        train, eval_rows, manifest = (
            content["train"],
            content["eval"],
            content["manifest"],
        )
        if not (
            isinstance(train, list)
            and isinstance(eval_rows, list)
            and isinstance(manifest, dict)
            and len(train) == header.get("train_rows")
            and len(eval_rows) == header.get("eval_rows")
        ):
            return None
        return train, eval_rows, manifest
    except FileNotFoundError:
        return None
    except Exception as error:  # noqa: BLE001 - any damage means re-prepare
        LOG.warning(
            "Decision prepared-row cache unreadable (%s: %s); re-preparing",
            type(error).__name__,
            error,
        )
        return None


def store(
    root: Path,
    key: str,
    identity_payload: Mapping[str, Any],
    cfg: Any,
    train: Sequence[Any],
    eval_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Write the canvas layer: ``train`` holds source pools (or premixed rows)."""
    parent = root / CACHE_DIR
    parent.mkdir(parents=True, exist_ok=True)
    path = _cache_path(root, key)
    content = {
        "train": list(train),
        "eval": list(eval_rows),
        "manifest": dict(manifest),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "wb", buffering=1 << 24) as handle:
            handle.write(MAGIC)
            writer = _HashingWriter(handle)
            pickle.dump(content, writer, protocol=5)
            header = json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "producer": PRODUCER,
                    "key": key,
                    "identity_sha256": _sha(identity_payload),
                    "config": _semantic_config(cfg),
                    "train_rows": len(content["train"]),
                    "eval_rows": len(content["eval"]),
                    "rows_length": writer.length,
                    "rows_sha256": writer.digest.hexdigest(),
                },
                sort_keys=True,
                default=_json_default,
            ).encode("utf-8")
            handle.write(header)
            handle.write(_TRAILER.pack(len(header), MAGIC))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException as error:
        Path(temporary).unlink(missing_ok=True)
        if not isinstance(error, (pickle.PicklingError, TypeError, AttributeError)):
            raise
        # A row the cache cannot hold costs a re-preparation next run, not this run.
        LOG.warning(
            "Decision prepared-row cache not written (%s: %s)",
            type(error).__name__,
            error,
        )
