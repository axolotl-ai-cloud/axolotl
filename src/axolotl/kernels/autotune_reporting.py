# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Bounded reporting and provenance for cached autotune selections."""

import hashlib
import json
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from transformers import TrainerCallback


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@lru_cache(maxsize=128)
def _source_digest(filename):
    try:
        return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def autotuner_metadata(autotuner, module, config_serializer=None):
    """Identify the source module and unpruned matrix without exposing source paths."""
    result = {}
    filename = getattr(module, "__file__", None)
    if filename and (digest := _source_digest(filename)):
        result["module_source_sha256"] = digest
    configs = getattr(autotuner, "configs", None)
    if isinstance(configs, (list, tuple)) and configs:
        matrix = [
            {
                "kwargs": config.kwargs,
                "num_warps": getattr(config, "num_warps", None),
                "num_stages": getattr(config, "num_stages", None),
                "num_ctas": getattr(config, "num_ctas", None),
            }
            for config in configs
        ]
        if config_serializer is not None:
            matrix = [config_serializer(config) for config in configs]
        result["candidate_configs_sha256"] = _fingerprint(matrix)
        result["candidate_config_count"] = len(matrix)
    return result


@lru_cache(maxsize=1)
def _runtime_versions():
    versions = {}
    for package in (
        "axolotl",
        "torch",
        "triton",
        "kernels",
        "quack-kernels",
        "sonic-moe",
        "nvidia-cutlass-dsl",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


class AutotuneTelemetryCallback(TrainerCallback):
    """Sample caches at bounded checkpoints and send each distinct record once.

    Source fingerprints describe loaded module files, not transitive dependencies.
    Candidate matrices describe the original search space, before runtime pruning.
    """

    event_type = "triton-autotune"
    poll_steps = (1, 2, 3, 4, 5, 25, 100)
    max_records_per_event = 128
    max_records_per_run = 1024

    def __init__(self):
        self._reported = False
        self._next_poll = 0
        self._seen: set[str] = set()

    def _collect_configs(self) -> list[dict]:
        raise NotImplementedError

    def _hardware_info(self) -> dict:
        return {}

    def _coverage_info(self) -> dict:
        return {}

    def _report(self, state, final=False):
        from axolotl.telemetry.manager import TelemetryManager

        manager = TelemetryManager.get_instance()
        if not manager.enabled:
            self._reported = True
            return
        records = []
        fingerprints = set()
        pending = 0
        limit = min(
            self.max_records_per_event, self.max_records_per_run - len(self._seen)
        )
        for record in self._collect_configs():
            fingerprint = _fingerprint(record)
            if fingerprint in self._seen or fingerprint in fingerprints:
                continue
            if len(records) >= limit:
                pending += 1
                continue
            records.append(record)
            fingerprints.add(fingerprint)
        coverage = self._coverage_info() if final else {}
        if records or coverage:
            manager.send_event(
                event_type=self.event_type,
                properties={
                    "kernel_count": len(records),
                    "kernels": records,
                    "global_step": state.global_step,
                    "final_snapshot": final,
                    "unreported_record_count": pending,
                    "runtime_versions": _runtime_versions(),
                    **self._hardware_info(),
                    **coverage,
                },
            )
            self._seen.update(fingerprints)
        if final or len(self._seen) >= self.max_records_per_run:
            self._reported = True

    def on_step_end(self, args, state, control, **kwargs):
        if self._reported or self._next_poll >= len(self.poll_steps):
            return
        if state.global_step < self.poll_steps[self._next_poll]:
            return
        while (
            self._next_poll < len(self.poll_steps)
            and state.global_step >= self.poll_steps[self._next_poll]
        ):
            self._next_poll += 1
        self._report(state)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._reported:
            self._report(state, final=True)
