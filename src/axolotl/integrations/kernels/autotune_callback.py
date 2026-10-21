# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Trainer callback for reporting selections from loaded Triton kernels."""

import torch

from axolotl.kernels.autotune_reporting import AutotuneTelemetryCallback


def _get_gpu_info() -> dict:
    """Return basic GPU identification for the current device."""
    if not torch.cuda.is_available():
        return {}
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_memory_bytes": props.total_memory,
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def _get_smem_capacity() -> dict:
    """Return shared memory capacity (bytes). Prefers the runtime lora_ops helper, falls
    back to the Triton active-driver device properties, then to torch device properties —
    so a dsv4-only run (no scattermoe) still reports smem."""
    # 1) lora_ops helper (matches what the scattermoe autotuner actually saw)
    try:
        from axolotl.integrations.kernels.autotune_collector import (
            _find_lora_ops_module,
        )

        lora_ops = _find_lora_ops_module()
        fn = getattr(lora_ops, "_get_smem_capacity", None) if lora_ops else None
        if fn is not None:
            return {"smem_capacity_bytes": fn()}
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    # 2) Triton active driver
    try:
        import triton

        idx = torch.cuda.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(idx)
        smem = props.get("max_shared_mem")
        if smem:
            return {"smem_capacity_bytes": int(smem)}
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    # 3) torch device properties
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        for name in (
            "shared_memory_per_block_optin",
            "shared_memory_per_multiprocessor",
        ):
            val = getattr(props, name, None)
            if val:
                return {"smem_capacity_bytes": int(val), "smem_source": name}
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return {}


class AutotuneReportCallback(AutotuneTelemetryCallback):
    """Report selections from loaded Triton kernels."""

    event_type = "triton-autotune"

    def _collect_configs(self):
        from axolotl.integrations.kernels.autotune_collector import (
            collect_autotune_configs,
        )

        self._coverage = []
        return collect_autotune_configs(coverage=self._coverage)

    def _coverage_info(self):
        coverage = getattr(self, "_coverage", [])
        return {"autotuner_coverage": coverage[:128]} if coverage else {}

    def _hardware_info(self):
        return {**_get_gpu_info(), **_get_smem_capacity()}
