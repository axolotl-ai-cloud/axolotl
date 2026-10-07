"""Shared config, JSON, hashing, and file helpers for decision training."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from axolotl.model_support import (
    DiffusionSpec,
    get_model_support_for_cfg,
    resolve_model_support,
)
from axolotl.utils.dict import DictDefault

CONTROL_TOKEN_FIELDS = (
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "unk_token_id",
    "mask_token_id",
)


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def model_config_overrides(cfg: Any) -> Any:
    """``model_config`` overrides, which validated configs store under the field name."""
    overrides = _value(cfg, "overrides_of_model_config")
    return _value(cfg, "model_config") if overrides is None else overrides


def require_diffusion_spec(cfg: Any) -> DiffusionSpec:
    if isinstance(cfg, Mapping) and not isinstance(cfg, DictDefault):
        cfg = DictDefault(cfg)
    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    spec = None if profile is None else profile.diffusion
    if not isinstance(spec, DiffusionSpec):
        raise ValueError("diffusion_decision requires a resolved native DiffusionSpec")
    return spec


def parse_token_ids(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    values = (
        (value,) if isinstance(value, int) and not isinstance(value, bool) else value
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an integer sequence")
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{name} must contain nonnegative integers")
    return tuple(values)


def optional_token_id(value: Any, name: str) -> int | None:
    token_ids = parse_token_ids(value, name)
    if len(token_ids) > 1:
        raise ValueError(f"{name} must resolve to at most one token ID")
    return token_ids[0] if token_ids else None


def required_token_id(value: Any, name: str) -> int:
    token_id = optional_token_id(value, name)
    if token_id is None:
        raise ValueError(f"{name} must resolve to exactly one token ID")
    return token_id


def resolve_mask_token_id(diffusion: Any, model_config: Any) -> int | None:
    value = _value(diffusion, "mask_token_id")
    if value is None:
        value = _value(model_config, "mask_token_id")
    return optional_token_id(value, "mask_token_id")


def active_control_ids(*sources: Any) -> set[int]:
    return {
        token_id
        for source in sources
        for name in CONTROL_TOKEN_FIELDS
        for token_id in parse_token_ids(_value(source, name), name)
    }


def canonical_json(value: Any, *, ensure_ascii: bool = False, **kwargs: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
        **kwargs,
    )


def canonical_sha256(value: Any, **kwargs: Any) -> str:
    return hashlib.sha256(canonical_json(value, **kwargs).encode("utf-8")).hexdigest()


def stable_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record at {path}:{line_number} must be an object"
                )
            records.append(record)
    return records


def atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
