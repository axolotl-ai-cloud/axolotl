# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Conservative forward search subsets from the October 2026 telemetry export.

The two shapes below have 29/108/21/49 runs on sm90/100/103/120.
Keep every winning tile family, including rare winners, and leave stages and
warps tunable: the export contains winners, not comparative timings or ranks.
"""

_FWD_SHAPES = {(1408, 2816), (2816, 704)}
_FWD_TILES = {
    (9, 0): {(64, 64, 64), (128, 64, 32)},
    (10, 0): {(64, 64, 64), (128, 64, 32)},
    (10, 3): {(64, 64, 64), (128, 64, 32), (64, 64, 32)},
    (12, 0): {
        (64, 64, 64),
        (128, 64, 32),
        (128, 32, 32),
        (32, 64, 64),
        (32, 32, 64),
    },
}
_FWD_LIMITS = {
    (9, 0): (232448, 1024, 524288),
    (10, 0): (232448, 2048, 655360),
    (10, 3): (232448, 1024, 393216),
    (12, 0): (101376, 1024, 524288),
}


def profile_fwd_configs(configs, capability, smem_capacity, meta):
    """Select tile families within the observed hardware/shape envelope.

    Unknown cases retain the resource-pruned matrix. Rank is absent from the
    historical data, so only small-rank BF16 workloads use this heuristic.
    """
    limits = _FWD_LIMITS.get(capability)
    if limits is None or (meta.get("N"), meta.get("K")) not in _FWD_SHAPES:
        return configs
    capacity, min_m, max_m = limits
    m_bucket = meta.get("M_BUCKET", 0)
    block_r = meta.get("BLOCK_R", 0)
    if (
        smem_capacity != capacity
        or not min_m <= m_bucket <= max_m
        or block_r not in (16, 32, 64)
    ):
        return configs
    for name in ("X_ptr", "W_ptr", "Y_ptr", "LA_ptr", "LB_ptr"):
        if str(getattr(meta.get(name), "dtype", None)) != "torch.bfloat16":
            return configs
    tiles = _FWD_TILES[capability]
    selected = [
        config
        for config in configs
        if tuple(config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        in tiles
    ]
    return selected or configs


_DX_MX_SHAPES = {
    (6144, 512): (1024, 8192),
    (1024, 6144): (1024, 8192),
    (6144, 2048): (8192, 262144),
    (4096, 6144): (8192, 262144),
}
_DX_MX_SPECS = {
    (32, 32, 32, 4, 2),
    (32, 32, 32, 4, 3),
    (32, 64, 32, 4, 2),
    (64, 64, 32, 4, 2),
}


def profile_dx_mx_configs(configs, capability, smem_capacity, meta):
    """Retain all Hopper MX dX winners across 30 runs and four sampled shapes.

    The tuple order is M, K, N, warps, stages. Historical B200/SM120 winners
    are outside the current safety matrix and must not be restored here.
    """
    bounds = _DX_MX_SHAPES.get((meta.get("N"), meta.get("K")))
    if (
        capability != (9, 0)
        or smem_capacity != 232448
        or bounds is None
        or not bounds[0] <= meta.get("M_BUCKET", 0) <= bounds[1]
        or meta.get("BLOCK_R") not in (16, 32, 64)
    ):
        return configs
    for name, dtype in (
        ("DY_ptr", "torch.bfloat16"),
        ("DX_ptr", "torch.bfloat16"),
        ("LA_ptr", "torch.bfloat16"),
        ("LB_ptr", "torch.bfloat16"),
        ("Wp_ptr", "torch.uint8"),
        ("Ws_ptr", "torch.float32"),
        ("Codebook_ptr", "torch.float32"),
    ):
        if str(getattr(meta.get(name), "dtype", None)) != dtype:
            return configs
    selected = [
        c
        for c in configs
        if (
            c.kwargs["BLOCK_M"],
            c.kwargs["BLOCK_K"],
            c.kwargs["BLOCK_N"],
            c.num_warps,
            c.num_stages,
        )
        in _DX_MX_SPECS
    ]
    return selected or configs
