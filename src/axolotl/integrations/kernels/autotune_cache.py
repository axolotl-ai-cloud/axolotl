# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Persist Triton autotune winners across runs and seed them before the first forward.

Triton keeps each ``@triton.autotune`` winner only in the process (``Autotuner.cache``), so a
restarted run re-benchmarks every config for every new (shape, dtype) key. With expert
parallelism that benchmark runs at different moments on different ranks, which the DeepEP
combine barrier reads as a straggler. Seeding the caches from the previous run's winners
removes the benchmark from the training step.

The file holds the records ``collect_autotune_configs`` produces, plus the GPU they were
measured on. A record is applied only to an autotuner whose candidate-config matrix matches
the one it was measured against.
"""

import json
import os
import sys
import tempfile
import time
from collections import Counter
from typing import Any

import torch
from transformers import TrainerCallback

from axolotl.integrations.kernels.autotune_collector import (
    _KERNEL_MODULE_HINTS,
    _config_to_dict,
    _is_autotuner,
    collect_autotune_configs,
)
from axolotl.kernels.autotune_reporting import autotuner_metadata
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

FILE_VERSION = 1
_CONFIG_META_KEYS = ("num_warps", "num_stages", "num_ctas")


def _gpu_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu_name": props.name,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
    }


def _record_identity(record: dict[str, Any]) -> tuple:
    return (
        record.get("module_fqn") or record.get("module"),
        record.get("kernel"),
        json.dumps(record.get("key"), sort_keys=True),
    )


def _key_tuple(autotuner: Any, key: dict[str, Any]) -> tuple:
    """Rebuild Triton's cache key: declared key args in order, then the dtype strings."""
    names = list(getattr(autotuner, "keys", None) or [])
    values = [key[name] for name in names if name in key]
    values.extend(str(v) for v in key.get("_extra", []))
    return tuple(values)


def _make_config(autotuner: Any, wanted: dict[str, Any]) -> Any:
    """Return the autotuner's own Config matching ``wanted``, or build an equivalent one."""
    normalized = {k: v for k, v in wanted.items() if v is not None}
    for config in getattr(autotuner, "configs", None) or []:
        have = {k: v for k, v in _config_to_dict(config).items() if v is not None}
        if have == normalized:
            return config
    from triton import Config

    kwargs = {k: v for k, v in normalized.items() if k not in _CONFIG_META_KEYS}
    return Config(
        kwargs,
        num_warps=normalized.get("num_warps", 4),
        num_stages=normalized.get("num_stages", 3),
        num_ctas=normalized.get("num_ctas", 1),
    )


def _autotuners() -> list[tuple[str, Any, Any]]:
    """All (module_fqn, module, autotuner) triples among the loaded kernel modules."""
    found = []
    for modname, module in list(sys.modules.items()):
        if module is None or not any(h in modname for h in _KERNEL_MODULE_HINTS):
            continue
        for attr in dir(module):
            obj = getattr(module, attr, None)
            if _is_autotuner(obj):
                found.append((modname, module, obj))
    return found


def _kernel_name(autotuner: Any) -> str:
    base = getattr(autotuner, "base_fn", None) or getattr(autotuner, "fn", None)
    return getattr(base, "__name__", "")


def read_autotune_cache(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("version") != FILE_VERSION:
        LOG.warning(
            "Ignoring autotune cache %s: unknown version %s", path, data.get("version")
        )
        return None
    return data


def seed_autotune_caches(records: list[dict[str, Any]]) -> tuple[int, Counter]:
    """Insert ``records`` into the matching loaded autotuners; returns (applied, skipped)."""
    skipped: Counter = Counter()
    applied = 0
    by_fqn: dict[tuple[Any, Any], list[tuple[Any, Any]]] = {}
    by_short: dict[tuple[Any, Any], list[tuple[Any, Any]]] = {}
    for modname, module, autotuner in _autotuners():
        kname = _kernel_name(autotuner)
        by_fqn.setdefault((modname, kname), []).append((module, autotuner))
        by_short.setdefault((modname.rsplit(".", 1)[-1], kname), []).append(
            (module, autotuner)
        )

    for record in records:
        kernel = record.get("kernel")
        targets = by_fqn.get((record.get("module_fqn"), kernel)) or by_short.get(
            (record.get("module"), kernel)
        )
        if not targets:
            skipped["kernel not loaded"] += 1
            continue
        for module, autotuner in targets:
            metadata = autotuner_metadata(autotuner, module)
            want_sha = record.get("candidate_configs_sha256")
            if want_sha and metadata.get("candidate_configs_sha256") != want_sha:
                skipped["config matrix changed"] += 1
                continue
            try:
                key = _key_tuple(autotuner, record["key"])
            except (KeyError, TypeError):
                skipped["malformed key"] += 1
                continue
            if key in autotuner.cache:
                skipped["already in cache"] += 1
                continue
            try:
                autotuner.cache[key] = _make_config(autotuner, record["config"])
            except Exception as exc:  # pylint: disable=broad-exception-caught
                skipped[f"config rejected ({type(exc).__name__})"] += 1
                continue
            applied += 1
    return applied, skipped


def load_autotune_cache(path: str) -> int:
    """Seed the loaded kernels from ``path`` when it was measured on this GPU."""
    data = read_autotune_cache(path)
    if data is None:
        LOG.info(
            "No autotune cache at %s; winners will be measured and written there", path
        )
        return 0
    gpu = _gpu_identity()
    if gpu and data.get("gpu_name") and data["gpu_name"] != gpu["gpu_name"]:
        LOG.warning(
            "Not seeding autotune cache %s: measured on %s, this run is on %s",
            path,
            data["gpu_name"],
            gpu["gpu_name"],
        )
        return 0
    applied, skipped = seed_autotune_caches(data.get("records", []))
    message = "Seeded %d autotune winner(s) from %s (%d in file%s)"
    args = (
        applied,
        path,
        len(data.get("records", [])),
        "".join(f"; skipped {n} {why}" for why, n in skipped.items()),
    )
    # a second pass (train start) finds the first pass's entries: not worth a line
    if applied == 0 and set(skipped) <= {"already in cache"}:
        LOG.debug(message, *args)
    else:
        LOG.info(message, *args)
    return applied


def gather_autotune_records() -> list[dict[str, Any]]:
    """This rank's winners, merged with every other rank's when distributed."""
    local = collect_autotune_configs()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world: list[list[dict[str, Any]] | None] = [
            None
        ] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(world, local)
        merged: dict[tuple, dict[str, Any]] = {}
        for rank_records in world:
            for rank_record in rank_records or []:
                merged.setdefault(_record_identity(rank_record), rank_record)
        return list(merged.values())
    return local


def write_autotune_cache(path: str, records: list[dict[str, Any]]) -> int:
    """Merge ``records`` into the file at ``path`` (newer wins) and write it atomically."""
    existing = read_autotune_cache(path)
    merged: dict[tuple, dict[str, Any]] = {}
    gpu = _gpu_identity()
    if existing and (not gpu or existing.get("gpu_name") == gpu.get("gpu_name")):
        for record in existing.get("records", []):
            merged[_record_identity(record)] = record
    for record in records:
        merged[_record_identity(record)] = record
    payload = {
        "version": FILE_VERSION,
        **gpu,
        "torch": torch.__version__,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": list(merged.values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=str)
    os.replace(tmp, path)
    return len(payload["records"])


def _is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


class AutotuneCacheCallback(TrainerCallback):
    """Seed again at train start (kernels imported lazily) and persist at each save and at the end."""

    def __init__(self, path: str):
        self.path = path

    def on_train_begin(self, args, state, control, **kwargs):
        load_autotune_cache(self.path)

    def _persist(self):
        records = gather_autotune_records()
        if _is_main_process() and records:
            count = write_autotune_cache(self.path, records)
            LOG.info("Wrote %d autotune winner(s) to %s", count, self.path)

    def on_save(self, args, state, control, **kwargs):
        self._persist()

    def on_train_end(self, args, state, control, **kwargs):
        self._persist()
