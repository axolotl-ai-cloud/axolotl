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
    return any(
        part in ("quack", "sonicmoe", "sonic_moe")
        or part.startswith(("sonic_moe_", "sonicmoe_"))
        for part in name.split(".")
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


def collect_sonicmoe_autotune_configs(coverage=None):
    """Collect Quack winners and any bundled Triton winners, deduplicating aliases.

    Quack signatures preserve declared scalar values followed by shape/stride/dtype
    triples. Older releases stringify these values; newer releases retain tuples.
    """
    results = []
    seen = set()
    for name, module in list(sys.modules.items()):
        if module is None or not _is_sonic_module(name):
            continue
        for obj in vars(module).values():
            if not _is_autotuner(obj) or id(obj) in seen:
                continue
            seen.add(id(obj))
            base = getattr(obj, "base_fn", None) or obj.fn
            kernel_name = getattr(base, "__name__", None)
            if not kernel_name:
                continue
            defining_name = getattr(base, "__module__", name)
            if not _is_sonic_module(defining_name):
                defining_name = name
            defining_module = sys.modules.get(defining_name, module)
            is_quack = any(part == "quack" for part in type(obj).__module__.split("."))
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
                        "key": _quack_key(obj, key)
                        if is_quack
                        else _label_key(obj, key),
                        "config": _plain(serialize(config)),
                        **metadata,
                    }
                except (TypeError, ValueError, AttributeError):
                    status["unsupported_entries"] += 1
                    continue
                results.append(record)
    return results


class SonicMoEAutotuneReportCallback(AutotuneTelemetryCallback):
    """Report external SonicMoE/Quack winners and final loaded-cache coverage."""

    event_type = "sonicmoe-autotune"

    def _collect_configs(self):
        self._coverage = []
        return collect_sonicmoe_autotune_configs(self._coverage)

    def _coverage_info(self):
        coverage = getattr(self, "_coverage", [])
        return {"autotuner_coverage": coverage[:128]} if coverage else {}

    def _hardware_info(self):
        return {**_get_gpu_info(), **_get_smem_capacity()}
