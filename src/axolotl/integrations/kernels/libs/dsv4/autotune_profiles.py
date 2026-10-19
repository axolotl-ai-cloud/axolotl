# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Backward subsets retaining all October 2026 telemetry winners.

B200 profiles cover 23 runs; GB10 profiles cover 27 runs. Unobserved shapes,
dtypes and hardware keep the caller's matrix. Config tuples are tile, warps,
stages. Default launch constants absent from the export are guarded explicitly.
"""

import torch

_PROFILES = {
    ("rmsln", (10, 0)): {(64, 4, 3), (64, 8, 3), (128, 8, 3), (256, 8, 3)},
    ("rmsln", (12, 1)): {(128, 8, 3)},
    ("collapse", (10, 0)): {(1024, 4, 3), (1024, 8, 3)},
    ("collapse", (12, 1)): {(256, 8, 3), (512, 4, 3), (1024, 4, 3), (1024, 8, 3)},
    ("pool", (10, 0)): {(32, 2, 3), (32, 4, 3), (32, 8, 3)},
}
_POINTERS = {
    "rmsln": ("STREAMS", "FN", "R", "G", "DMIX", "DSTREAMS", "DFN"),
    "collapse": ("PRE", "STREAMS", "GOUT", "DPRE", "DSTREAMS"),
    "pool": ("KV", "GATE", "DOUT", "DKV", "DGATE"),
}


def profile_bwd_configs(configs, kernel, capability, meta):
    """Filter existing configs only, falling back on any uncovered input."""
    specs = _PROFILES.get((kernel, capability))
    if specs is None:
        return configs
    m = meta.get("M", 0)
    if capability == (12, 1):
        if m not in (64, 128):
            return configs
    else:
        lower, upper = (1, 8192) if kernel == "pool" else (128, 32768)
        if not lower <= m <= upper:
            return configs
    if kernel == "rmsln":
        if (meta.get("K"), meta.get("N"), meta.get("BM")) != (16384, 24, 16):
            return configs
        tile = "BK"
    elif kernel == "collapse":
        if (meta.get("D"), meta.get("HC")) != (4096, 4):
            return configs
        tile = "BD"
    else:
        if meta.get("D") != 512 or meta.get("W") not in (8, 128):
            return configs
        tile = "BD"
    for name in _POINTERS[kernel]:
        dtype = (
            "torch.bfloat16"
            if kernel == "pool" and name in ("KV", "GATE")
            else "torch.float32"
        )
        if str(getattr(meta.get(name), "dtype", None)) != dtype:
            return configs
    selected = [
        c for c in configs if (c.kwargs[tile], c.num_warps, c.num_stages) in specs
    ]
    return selected or configs


def bwd_pruner(kernel):
    """Build an early-prune hook using the input tensor's CUDA device."""

    def prune(configs, named_args, **kwargs):
        meta = {**named_args, **kwargs}
        tensor = meta.get(_POINTERS[kernel][0])
        if torch.version.hip or tensor is None or tensor.device.type != "cuda":
            return configs
        return profile_bwd_configs(
            configs, kernel, torch.cuda.get_device_capability(tensor.device), meta
        )

    return prune
