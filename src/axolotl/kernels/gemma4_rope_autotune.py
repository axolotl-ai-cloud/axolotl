# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Backward RMSNorm/RoPE search subsets from October 2026 telemetry.

Each profile has at least 100 BF16 observations and retains every observed
winner. Groups needing more than eight of the twelve configs keep the full
matrix: their winners are too dispersed to justify a narrow search.
"""

import torch

_BWD_PROFILES = {
    ((8, 0), 256): {(2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3), (8, 3)},
    ((8, 0), 512): {(2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3), (8, 3)},
    ((8, 6), 256): {(2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)},
    ((8, 9), 256): {(2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3), (8, 2)},
    ((9, 0), 256): {
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 1),
        (4, 2),
        (4, 3),
        (8, 3),
        (16, 2),
    },
    ((10, 0), 256): {(2, 1), (2, 2), (2, 3), (4, 1)},
    ((10, 0), 512): {(2, 1), (2, 2), (4, 1), (4, 2), (4, 3), (8, 1)},
    ((10, 3), 256): {(2, 1), (2, 2), (2, 3)},
    ((10, 3), 512): {(2, 2), (4, 1), (4, 2), (4, 3)},
    ((12, 0), 256): {(2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)},
}


def profile_rope_bwd_configs(configs, capability, meta):
    """Return a BF16 profile, or the original matrix for uncovered inputs.

    The caller must check the backend: HIP capabilities can overlap CUDA SMs.
    """
    specs = _BWD_PROFILES.get((capability, meta.get("n_cols")))
    if specs is None or not meta.get("HAS_WEIGHT"):
        return configs
    for name in ("dY_ptr", "dX_ptr", "X_ptr", "W_ptr", "COS_ptr", "SIN_ptr"):
        if str(getattr(meta.get(name), "dtype", None)) != "torch.bfloat16":
            return configs
    for name in ("RSTD_ptr", "dW_ptr"):
        if str(getattr(meta.get(name), "dtype", None)) != "torch.float32":
            return configs
    selected = [
        config for config in configs if (config.num_warps, config.num_stages) in specs
    ]
    return selected or configs


def prune_rope_bwd_configs(configs, named_args, **kwargs):
    """Apply profiles on the input's CUDA device, merging launch metadata."""
    meta = {**named_args, **kwargs}
    x = meta.get("X_ptr")
    if torch.version.hip or x is None or x.device.type != "cuda":
        return configs
    return profile_rope_bwd_configs(
        configs, torch.cuda.get_device_capability(x.device), meta
    )
