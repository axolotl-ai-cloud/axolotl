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
import math
import os
import sys
import tempfile
import time
from collections import Counter
from importlib import metadata
from typing import Any, Callable

import torch
from transformers import TrainerCallback

from axolotl.integrations.kernels.autotune_collector import (
    _KERNEL_MODULE_HINTS,
    _config_to_dict,
    _is_autotuner,
    collect_autotune_configs,
)
from axolotl.integrations.kernels.sonicmoe_autotune import (
    _plain,
    _quack_config,
    collect_sonicmoe_autotune_configs,
    collect_sonicmoe_records,
    quack_file_metadata,
    sonicmoe_autotuners,
)
from axolotl.kernels.autotune_reporting import _fingerprint, autotuner_metadata
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


class _TimingCache(dict):
    """An autotuner ``cache`` that also keeps the full benchmark table of every key it stores.

    Triton and quack keep only the last table (``configs_timings``) and assign it right after
    the winner goes into ``cache[key]``, both after a live benchmark and on a disk-cache hit, so
    a key's table is picked up when the next key is stored or when the records are read.
    ``bench_time`` (the sweep's wall time, compiles included) is set just before ``cache[key]``
    by a benchmark and not refreshed by a disk-cache hit, so it counts only when it changed.
    """

    def __init__(self, tuner: Any, entries: dict):
        super().__init__(entries)
        self.tuner = tuner
        self.timings: dict[tuple, dict] = {}
        self.sweep_seconds: dict[tuple, float] = {}
        self.call_args: dict[tuple, dict] = {}
        self.candidate_counts: dict[tuple, int] = {}
        self.candidate_shas: dict[tuple, str] = {}
        self.seeded: set[tuple] = set()
        # seeded key -> the swept-candidate fingerprint on file, until a call checks it
        self.unverified: dict[tuple, str] = {}
        self.profiled: set[tuple] = set()
        self.pinned: set[tuple] = set()
        self.heuristic: set[tuple] = set()
        self.origins: dict[tuple, tuple] = {}
        self.aliases: dict[tuple, tuple] = {}
        self.exact_fallbacks: set[tuple] = set()
        self._pending: tuple | None = None
        self._pending_sweep: float | None = None
        self._last_table = getattr(tuner, "configs_timings", None)
        self._last_bench = getattr(tuner, "bench_time", None)

    def _policy(self):
        return getattr(self.tuner, "_axolotl_key_policy", None)

    def __contains__(self, key):
        policy = self._policy()
        if policy is None or not policy.in_call():
            return super().__contains__(key)
        return policy.lookup(self, key)

    def __getitem__(self, key):
        policy = self._policy()
        return super().__getitem__(key if policy is None else policy.storage_for(key))

    def __setitem__(self, key, value):
        policy = self._policy()
        if policy is not None:
            key = policy.storage_for(key)
            policy.stored(self, key)
        self.harvest()
        self._pending = key
        tuner = self.tuner
        bench = getattr(tuner, "bench_time", None)
        changed = isinstance(bench, (int, float)) and bench != self._last_bench
        self._pending_sweep = float(bench) if changed else None
        self._last_bench = bench
        if getattr(tuner, "_axolotl_profile_hit", False):
            self.profiled.add(key)
            tuner._axolotl_profile_hit = False
        if getattr(tuner, "_axolotl_skip_bench", False):
            self.pinned.add(key)
            tuner._axolotl_skip_bench = False
        noted = getattr(tuner, "_axolotl_call_args", None)
        if noted is not None:
            self.call_args[key] = noted
            tuner._axolotl_call_args = None
        count = getattr(tuner, "_axolotl_candidate_count", None)
        if count is not None:
            self.candidate_counts[key] = count
            tuner._axolotl_candidate_count = None
        sha = getattr(tuner, "_axolotl_candidate_sha256", None)
        if sha is not None:
            self.candidate_shas[key] = sha
            tuner._axolotl_candidate_sha256 = None
        super().__setitem__(key, value)

    def harvest(self) -> None:
        table = getattr(self.tuner, "configs_timings", None)
        pending = self._pending
        if (
            pending is not None
            and pending not in self.pinned
            and table is not None
            and table is not self._last_table
        ):
            self.timings[pending] = dict(table)
            if self._pending_sweep is not None:
                self.sweep_seconds[pending] = self._pending_sweep
        self._last_table = table
        self._pending = None


def note_call_args(tuner: Any, kwargs: dict[str, Any]) -> None:
    """Remember the shapes and declared key values of the call about to be tuned.

    quack's cache key lists tensor shapes without argument names; these let a key be
    matched back by name (``autotune_profiles`` ``match``).
    """
    names = getattr(tuner, "arg_names", None)
    keys = set(getattr(tuner, "keys", None) or ())
    noted: dict[str, Any] = {}
    for name, value in {**(getattr(tuner, "nargs", None) or {}), **kwargs}.items():
        if names is not None and name not in names:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and hasattr(value, "dtype"):
            noted[f"{name}.shape"] = [int(d) for d in shape]
        elif name in keys:
            try:
                noted[name] = _plain(value)
            except TypeError:
                noted[name] = str(value)
    tuner._axolotl_call_args = noted


def quack_candidates_sha256(configs: list) -> str | None:
    """Order-independent fingerprint of a quack candidate list, or None if not serializable."""
    try:
        items = sorted(
            json.dumps(_plain(_quack_config(c)), sort_keys=True, default=str)
            for c in configs
        )
    except (TypeError, ValueError, AttributeError):
        return None
    return _fingerprint(items)


def note_candidates(tuner: Any, configs: list) -> None:
    """Remember the size and fingerprint of the candidate list this call will sweep."""
    tuner._axolotl_candidate_count = len(configs)
    tuner._axolotl_candidate_sha256 = quack_candidates_sha256(configs)


def _note_quack_calls(tuner: Any) -> None:
    if getattr(tuner, "_axolotl_notes_calls", False):
        return
    orig_prune = tuner.prune_configs

    def prune_configs(kwargs):
        note_call_args(tuner, kwargs)
        configs = orig_prune(kwargs)
        note_candidates(tuner, configs)
        return configs

    tuner.prune_configs = prune_configs
    tuner._axolotl_notes_calls = True


def install_timing_capture() -> int:
    """Keep every benchmarked config's time per key on each loaded autotuner. Idempotent.

    Covers the Triton kernel modules and the SonicMoE/quack ones, so the cache files carry the
    full distribution, not just the winner. Kernel modules imported after this call are picked
    up by a later call.
    """
    installed = 0
    tuners = [(tuner, False) for _name, _module, tuner in _autotuners()]
    tuners += [(tuner, quack) for _name, _module, tuner, quack in sonicmoe_autotuners()]
    for autotuner, is_quack in tuners:
        if is_quack and hasattr(autotuner, "prune_configs"):
            _note_quack_calls(autotuner)
        if isinstance(autotuner.cache, _TimingCache):
            continue
        autotuner.cache = _TimingCache(autotuner, autotuner.cache)
        installed += 1
    return installed


def _timing_table(table: dict, serialize: Any) -> tuple[dict[str, float], int]:
    timings: dict[str, float] = {}
    failed = 0
    for config, value in table.items():
        ms = value[0] if isinstance(value, (list, tuple)) else value
        try:
            ms = float(ms)
        except (TypeError, ValueError):
            failed += 1
            continue
        if not math.isfinite(ms):
            failed += 1
            continue
        try:
            name = json.dumps(serialize(config), sort_keys=True, default=str)
        except (TypeError, ValueError, AttributeError):
            continue
        timings[name] = ms
    return dict(sorted(timings.items(), key=lambda kv: kv[1])), failed


def timing_fields(
    autotuner: Any, key_tuple: tuple, serialize: Any = _config_to_dict
) -> dict[str, Any]:
    """How ``key_tuple``'s winner was chosen, with its benchmark table when one was recorded."""
    cache = autotuner.cache
    if not isinstance(cache, _TimingCache):
        return {}
    cache.harvest()
    out: dict[str, Any] = {}
    if key_tuple in cache.call_args:
        out["call_args"] = cache.call_args[key_tuple]
    if key_tuple in cache.candidate_counts:
        out["candidate_config_count"] = cache.candidate_counts[key_tuple]
    if key_tuple in cache.candidate_shas:
        out["swept_configs_sha256"] = cache.candidate_shas[key_tuple]
    reuses = sum(1 for bucket, _w in cache.aliases.values() if bucket == key_tuple)
    if reuses:
        out["bucket_reuses"] = reuses
    if key_tuple in cache.exact_fallbacks:
        out["bucket_fallback"] = True
    if key_tuple in cache.seeded:
        return {**out, "source": "seeded"}
    if key_tuple in cache.heuristic:
        return {**out, "source": "heuristic"}
    table = cache.timings.get(key_tuple)
    if key_tuple in cache.profiled:
        out["source"] = "profile"
    elif table is not None:
        out["source"] = "measured"
    else:
        return {**out, "source": "untimed"}
    if table is not None:
        timings, failed = _timing_table(table, serialize)
        out.update(
            timings_ms=timings, configs_benchmarked=len(table), failed_configs=failed
        )
    if key_tuple in cache.sweep_seconds:
        out["sweep_seconds"] = round(cache.sweep_seconds[key_tuple], 3)
    return out


def _kernel_name(autotuner: Any) -> str:
    base = getattr(autotuner, "base_fn", None) or getattr(autotuner, "fn", None)
    return getattr(base, "__name__", "")


def read_autotune_cache(path: str) -> dict[str, Any] | None:
    """The cache file's contents, or None when absent, unreadable, or of another version.

    The cache is an optimization: a truncated file (a run killed mid-write before the
    atomic rename, or a copy that was cut short) must not abort a run, and the next
    write repairs it.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        LOG.warning("Ignoring unreadable autotune cache %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        LOG.warning("Ignoring autotune cache %s: not a JSON object", path)
        return None
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
            if isinstance(autotuner.cache, _TimingCache):
                autotuner.cache.seeded.add(key)
            applied += 1
    return applied, skipped


def load_autotune_cache(path: str) -> int:
    """Seed the loaded kernels from ``path`` when it was measured on this GPU."""
    install_timing_capture()
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


def seed_quack_caches(records: list[dict[str, Any]]) -> tuple[int, Counter]:
    """Insert measured quack winners whose exact key can be rebuilt; returns (applied, skipped).

    Only string keys round-trip through JSON exactly, which is what quack <= 0.6.1 builds (the
    vendored copy in the hub sonic-moe kernel); later releases keep tuples and dtypes.

    A record's ``swept_configs_sha256`` (the candidates left after profiles and pruning) depends
    on the call, so it is checked at the key's first lookup (``QuackKeyPolicy``): a winner swept
    over other candidates than this run's is re-tuned.
    """
    skipped: Counter = Counter()
    applied = 0
    targets: dict[tuple[Any, Any], list[tuple[Any, Any]]] = {}
    for name, module, tuner, is_quack in sonicmoe_autotuners():
        if is_quack:
            short = (_kernel_name(tuner), name.rsplit(".", 1)[-1])
            targets.setdefault(short, []).append((module, tuner))
    for record in records:
        if record.get("backend") != "quack" or record.get("source") != "measured":
            skipped["not a measured quack winner"] += 1
            continue
        found = targets.get((record.get("kernel"), record.get("module")))
        if not found:
            skipped["kernel not loaded"] += 1
            continue
        signature = (record.get("key") or {}).get("signature")
        if not isinstance(signature, list) or not all(
            isinstance(v, str) for v in signature
        ):
            skipped["key not rebuildable"] += 1
            continue
        key = tuple(signature)
        for module, tuner in found:
            try:
                metadata = autotuner_metadata(tuner, module, _quack_config)
            except (TypeError, ValueError, AttributeError):
                metadata = {}
            want_sha = record.get("candidate_configs_sha256")
            if want_sha and metadata.get("candidate_configs_sha256") != want_sha:
                skipped["config matrix changed"] += 1
                continue
            if key in tuner.cache:
                skipped["already in cache"] += 1
                continue
            config = None
            for candidate in tuner.configs:
                try:
                    if _plain(_quack_config(candidate)) == record.get("config"):
                        config = candidate
                        break
                except TypeError:
                    continue
            if config is None:
                skipped["config not a candidate"] += 1
                continue
            tuner.cache[key] = config
            if isinstance(tuner.cache, _TimingCache):
                tuner.cache.seeded.add(key)
                swept = record.get("swept_configs_sha256")
                if isinstance(swept, str):
                    tuner.cache.unverified[key] = swept
            applied += 1
    return applied, skipped


def load_quack_cache(path: str, quiet: bool = False) -> int:
    """Seed the loaded quack autotuners from the SonicMoE cache file at ``path``."""
    data = read_autotune_cache(path)
    if data is None:
        return 0
    gpu = _gpu_identity()
    if gpu and data.get("gpu_name") and data["gpu_name"] != gpu["gpu_name"]:
        if not quiet:
            LOG.warning(
                "Not seeding quack autotune cache %s: measured on %s, this run is on %s",
                path,
                data["gpu_name"],
                gpu["gpu_name"],
            )
        return 0
    applied, skipped = seed_quack_caches(data.get("records", []))
    if applied or not quiet:
        LOG.info(
            "Seeded %d quack autotune winner(s) from %s%s",
            applied,
            path,
            "".join(f"; skipped {n} {why}" for why, n in skipped.items()),
        )
    return applied


def _with_timings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[Any, Any], Any] = {}
    for modname, _module, autotuner in _autotuners():
        index[(modname, _kernel_name(autotuner))] = autotuner
    for record in records:
        autotuner = index.get((record.get("module_fqn"), record.get("kernel")))
        if autotuner is None:
            continue
        try:
            key = _key_tuple(autotuner, record["key"])
        except (KeyError, TypeError):
            continue
        record.update(timing_fields(autotuner, key))
    return records


def collect_triton_records() -> list[dict[str, Any]]:
    return _with_timings(collect_autotune_configs())


def gather_autotune_records(
    collect: Callable[[], list[dict[str, Any]]] = collect_triton_records,
) -> list[dict[str, Any]]:
    """This rank's winners, merged with every other rank's when distributed.

    A key benchmarked (or re-tuned) on one rank and seeded on another keeps the benchmarked
    record.
    """
    local = collect()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local
    world: list[list[dict[str, Any]] | None] = [
        None
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(world, local)
    merged: dict[tuple, dict[str, Any]] = {}
    for rank_records in world:
        for rank_record in rank_records or []:
            identity = _record_identity(rank_record)
            have = merged.get(identity)
            if (
                have is None
                or (not have.get("timings_ms") and rank_record.get("timings_ms"))
                or (
                    have.get("source") == "seeded"
                    and rank_record.get("source") != "seeded"
                )
            ):
                merged[identity] = rank_record
    return list(merged.values())


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _merge_record(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """A seeded or untimed re-collection of a key must not drop the measurement on file.

    A key swept over another candidate set (profiles changed) replaces the record either way.
    """
    if old is None:
        return new
    if new.get("source") == "seeded":
        return old
    swept = new.get("swept_configs_sha256")
    if swept and swept != old.get("swept_configs_sha256"):
        return new
    if (
        not new.get("timings_ms")
        and old.get("timings_ms")
        and old.get("config") == new.get("config")
    ):
        return old
    return new


def write_autotune_cache(
    path: str,
    records: list[dict[str, Any]],
    packages: tuple[str, ...] = ("triton",),
    extra: dict[str, Any] | None = None,
) -> int:
    """Merge ``records`` into the file at ``path`` (newer wins) and write it atomically."""
    existing = read_autotune_cache(path)
    merged: dict[tuple, dict[str, Any]] = {}
    gpu = _gpu_identity()
    if existing and (not gpu or existing.get("gpu_name") == gpu.get("gpu_name")):
        for record in existing.get("records", []):
            merged[_record_identity(record)] = record
    for record in records:
        identity = _record_identity(record)
        merged[identity] = _merge_record(merged.get(identity), record)
    payload = {
        "version": FILE_VERSION,
        **gpu,
        "torch": torch.__version__,
        "packages": package_versions(packages),
        **(extra or {}),
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


def quack_cache_path(path: str) -> str:
    """Where the SonicMoE/quack winners go: ``b200.json`` -> ``b200.quack.json``."""
    return f"{os.path.splitext(path)[0]}.quack.json"


def _is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


class AutotuneCacheCallback(TrainerCallback):
    """Seed again at train start (kernels imported lazily) and persist new winners early.

    Winners appear during the first steps (and whenever a new routing-imbalanced bucket shows
    up), so the file is written as soon as this rank's cache grows during the first
    ``early_steps`` steps, then at every save and at train end. With ``sonicmoe`` the
    SonicMoE/quack winners go to ``quack_cache_path(path)`` on the same schedule.
    """

    def __init__(
        self,
        path: str | None,
        profiles: list[dict[str, Any]] | None = None,
        early_steps: int = 20,
        neighborhood: int | None = None,
        sonicmoe: bool = False,
        quack_policy: tuple[str, bool] | None = None,
    ):
        self.path = path
        self.quack_policy = quack_policy
        self.profiles = profiles
        self.neighborhood = neighborhood
        self.early_steps = early_steps
        self.sonicmoe = sonicmoe
        self._last_count = -1

    def on_train_begin(self, args, state, control, **kwargs):
        if self.path:
            load_autotune_cache(self.path)
            if self.sonicmoe:
                load_quack_cache(quack_cache_path(self.path))
        if self.profiles:
            from axolotl.integrations.kernels.autotune_profiles import (
                apply_autotune_profiles,
            )

            apply_autotune_profiles(self.profiles)
        if self.neighborhood:
            from axolotl.integrations.kernels.autotune_neighborhood import (
                install_neighborhood_search,
            )

            install_neighborhood_search(self.neighborhood)
        self._install_quack_policy()

    def _install_quack_policy(self):
        if self.quack_policy is None:
            return
        from axolotl.integrations.kernels.quack_autotune_policy import (
            install_quack_key_policy,
        )

        install_quack_key_policy(*self.quack_policy)

    def _local_count(self) -> int:
        count = len(collect_autotune_configs())
        if self.sonicmoe:
            count += len(collect_sonicmoe_autotune_configs())
        return count

    def _persist(self):
        if not self.path:
            return
        records = gather_autotune_records()
        if _is_main_process() and records:
            count = write_autotune_cache(self.path, records)
            LOG.info("Wrote %d autotune winner(s) to %s", count, self.path)
        if self.sonicmoe:
            records = gather_autotune_records(collect_sonicmoe_records)
            if _is_main_process() and records:
                quack_path = quack_cache_path(self.path)
                packages, extra = quack_file_metadata()
                count = write_autotune_cache(quack_path, records, packages, extra)
                LOG.info(
                    "Wrote %d SonicMoE autotune winner(s) to %s", count, quack_path
                )
        self._last_count = self._local_count()

    def on_step_begin(self, args, state, control, **kwargs):
        # SonicMoE's quack autotuners can be imported by the first forward
        if state.global_step > self.early_steps:
            return
        if self.path:
            install_timing_capture()
            if self.sonicmoe:
                load_quack_cache(quack_cache_path(self.path), quiet=True)
        if self.profiles:
            from axolotl.integrations.kernels.autotune_profiles import (
                apply_autotune_profiles,
            )

            apply_autotune_profiles(self.profiles, quiet=True)
        self._install_quack_policy()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > self.early_steps or not self.path:
            return
        # every rank must agree on whether to run the collective gather
        grew = self._local_count() != self._last_count
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor(
                [int(grew)], device="cuda" if torch.cuda.is_available() else "cpu"
            )
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            grew = bool(flag.item())
        if grew:
            self._persist()

    def on_save(self, args, state, control, **kwargs):
        self._persist()

    def on_train_end(self, args, state, control, **kwargs):
        self._persist()
