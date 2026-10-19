# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Read loaded SonicMoE/Quack autotuners without importing CUDA kernels."""

import sys
from dataclasses import fields, is_dataclass
from enum import Enum

import torch

from axolotl.integrations.kernels.autotune_callback import (
    _get_gpu_info,
    _get_smem_capacity,
)
from axolotl.integrations.kernels.autotune_collector import (
    _config_to_dict,
    _is_autotuner,
    _label_key,
)
from axolotl.kernels.autotune_reporting import (
    AutotuneTelemetryCallback,
    autotuner_metadata,
)


def _is_sonic_module(name):
    # hub kernels load under a hash-named package such as ``_sonic_moe_cuda_89d4054``
    return any(
        part in ("quack", "sonicmoe", "sonic_moe")
        or part.startswith(("sonic_moe_", "sonicmoe_"))
        for part in (piece.lstrip("_") for piece in name.split("."))
    )


def _plain(value):
    if isinstance(value, Enum):
        return _plain(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    raise TypeError("Unsupported autotune metadata value")


def _quack_config(config):
    return _plain(config.kwargs)


def _quack_key(autotuner, key):
    # Optional key arguments may be omitted; their names cannot be recovered from the cache.
    return {
        "declared_names": list(autotuner.keys),
        "signature": _plain(key),
    }


QUACK_PACKAGES = ("quack-kernels", "nvidia-cutlass-dsl", "sonic-moe", "triton")


def loaded_quack_versions():
    """``{module name: {version, file}}`` for each loaded quack package, vendored ones included.

    The hub sonic-moe kernel ships its own quack, so the installed ``quack-kernels``
    distribution is not necessarily the one that ran.
    """
    found = {}
    for name, module in list(sys.modules.items()):
        if module is None or name.rsplit(".", 1)[-1] != "quack":
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            continue
        found[name] = {
            "version": str(version),
            "file": getattr(module, "__file__", None),
        }
    return found


def quack_file_metadata():
    """Versions for the quack cache file; drops ``quack-kernels`` when only a vendored quack ran."""
    loaded = loaded_quack_versions()
    packages = QUACK_PACKAGES
    if loaded and "quack" not in loaded:
        packages = tuple(p for p in packages if p != "quack-kernels")
    return packages, {"quack_loaded": loaded}


def sonicmoe_autotuners():
    """(defining module name, defining module, autotuner, is_quack) per loaded autotuner."""
    found = []
    seen = set()
    for name, module in list(sys.modules.items()):
        if module is None or not _is_sonic_module(name):
            continue
        for obj in list(vars(module).values()):
            if not _is_autotuner(obj) or id(obj) in seen:
                continue
            seen.add(id(obj))
            base = getattr(obj, "base_fn", None) or obj.fn
            if not getattr(base, "__name__", None):
                continue
            defining_name = getattr(base, "__module__", name)
            if not _is_sonic_module(defining_name):
                defining_name = name
            is_quack = any(part == "quack" for part in type(obj).__module__.split("."))
            found.append(
                (defining_name, sys.modules.get(defining_name, module), obj, is_quack)
            )
    return found


def collect_sonicmoe_autotune_configs(coverage=None, with_timings=False):
    """Collect Quack winners and any bundled Triton winners, deduplicating aliases.

    Quack signatures preserve declared scalar values followed by shape/stride/dtype
    triples. Older releases stringify these values; newer releases retain tuples.
    """
    if with_timings:
        from axolotl.integrations.kernels.autotune_cache import timing_fields
    results = []
    for defining_name, defining_module, obj, is_quack in sonicmoe_autotuners():
        base = getattr(obj, "base_fn", None) or obj.fn
        kernel_name = base.__name__
        backend = "quack" if is_quack else "triton"
        status = {
            "kernel": kernel_name,
            "module_fqn": defining_name,
            "backend": backend,
            "cache_entries": len(obj.cache),
            "status": "cache_populated" if obj.cache else "cache_empty",
            "unsupported_entries": 0,
        }
        if coverage is not None:
            coverage.append(status)
        if not obj.cache:
            continue
        serialize = _quack_config if is_quack else _config_to_dict
        try:
            metadata = autotuner_metadata(obj, defining_module, serialize)
        except (TypeError, ValueError, AttributeError):
            metadata = {}
        for key, config in list(obj.cache.items()):
            try:
                record = {
                    "kernel": kernel_name,
                    "module": defining_name.rsplit(".", 1)[-1],
                    "module_fqn": defining_name,
                    "backend": backend,
                    "key": _quack_key(obj, key) if is_quack else _label_key(obj, key),
                    "config": _plain(serialize(config)),
                    **metadata,
                }
                if with_timings:
                    record.update(timing_fields(obj, key, serialize))
            except (TypeError, ValueError, AttributeError):
                status["unsupported_entries"] += 1
                continue
            results.append(record)
    return results


def collect_sonicmoe_records():
    return collect_sonicmoe_autotune_configs(with_timings=True)


def _timing_spread(ms):
    """Best/worst time and the regret (% over the best) of the runner-up and the slowest
    candidate, over the whole benchmarked set (``ms`` sorted ascending)."""
    best, worst = ms[0], ms[-1]
    if best <= 0:
        return {}
    spread = {
        "best_ms": round(best, 4),
        "worst_ms": round(worst, 4),
        "worst_regret_pct": round((worst / best - 1) * 100, 2),
    }
    if len(ms) > 1:
        spread["runner_up_regret_pct"] = round((ms[1] / best - 1) * 100, 2)
    return spread


class SonicMoEAutotuneReportCallback(AutotuneTelemetryCallback):
    """Report external SonicMoE/Quack winners and final loaded-cache coverage."""

    event_type = "sonicmoe-autotune"
    # quack sweeps 40-80 candidates per key; the fastest few carry the profile signal
    max_timings_per_record = 8

    def _collect_configs(self):
        self._coverage = []
        records = collect_sonicmoe_autotune_configs(self._coverage, with_timings=True)
        for record in records:
            timings = record.get("timings_ms")
            if timings:
                record.update(_timing_spread(list(timings.values())))
            if timings and len(timings) > self.max_timings_per_record:
                kept = list(timings.items())[: self.max_timings_per_record]
                record["timings_ms"] = dict(kept)
                record["timings_ms_dropped"] = len(timings) - len(kept)
        return records

    def _coverage_info(self):
        coverage = getattr(self, "_coverage", [])
        return {"autotuner_coverage": coverage[:128]} if coverage else {}

    def _hardware_info(self):
        return {**_get_gpu_info(), **_get_smem_capacity()}
