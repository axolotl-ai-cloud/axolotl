# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Telemetry for the fused RMSNorm+RoPE Triton autotune selections.

Mirrors the scattermoe-lora autotune telemetry
(:mod:`axolotl.integrations.kernels.autotune_callback`): after the kernel's
``@triton.autotune`` cache is populated by the first backward pass, report the
selected configs alongside GPU identity so the per-hardware tuning that varies
across architectures can be aggregated.
"""

import torch

from axolotl.kernels.autotune_reporting import (
    AutotuneTelemetryCallback,
    autotuner_metadata,
)

# (human-readable name, attribute on gemma4_fused_rope, autotune key arg names)
_KERNEL_REGISTRY: list[tuple[str, str, list[str]]] = [
    ("fused_rms_norm_rope_bwd", "_rms_norm_rope_backward_kernel", ["n_cols"]),
]


def _get_gpu_info() -> dict:
    """Return basic GPU identification for the current device."""
    if not torch.cuda.is_available():
        return {}
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_memory_bytes": props.total_memory,
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def collect_fused_rope_autotune_configs() -> list[dict]:
    """Read the autotune ``.cache`` from the fused RMSNorm+RoPE backward kernel.

    Each entry is ``{"kernel", "key", "config"}`` — the same shape the
    scattermoe collector emits, so both event types aggregate uniformly.
    Returns ``[]`` if Triton/the kernel isn't loaded or nothing autotuned yet.
    """
    import sys

    # The kernel module is only in sys.modules once the fused path has run —
    # which is exactly when its autotune cache is populated. Read it from there
    # instead of importing (avoids pulling in Triton when the path is unused).
    mod = sys.modules.get("axolotl.kernels.gemma4_fused_rope")
    if mod is None:
        return []

    results: list[dict] = []
    for friendly_name, attr_name, key_names in _KERNEL_REGISTRY:
        kernel_fn = getattr(mod, attr_name, None)
        cache = getattr(kernel_fn, "cache", None)
        if not cache:
            continue
        key_names = list(getattr(kernel_fn, "keys", None) or key_names)
        metadata = autotuner_metadata(kernel_fn, mod)
        for key_tuple, config in cache.items():
            config_dict = dict(config.kwargs)
            config_dict["num_warps"] = config.num_warps
            config_dict["num_stages"] = config.num_stages
            if getattr(config, "num_ctas", None) is not None:
                config_dict["num_ctas"] = config.num_ctas

            key: dict = {}
            for i, name in enumerate(key_names):
                if i < len(key_tuple):
                    key[name] = key_tuple[i]
            if len(key_tuple) > len(key_names):
                key["_extra"] = [str(v) for v in key_tuple[len(key_names) :]]

            results.append(
                {"kernel": friendly_name, "key": key, "config": config_dict, **metadata}
            )
    return results


class FusedRopeAutotuneReportCallback(AutotuneTelemetryCallback):
    """Report fused RMSNorm+RoPE selections, including later cache entries."""

    event_type = "fused-rope-autotune"

    def _collect_configs(self):
        return collect_fused_rope_autotune_configs()

    def _hardware_info(self):
        return _get_gpu_info()
