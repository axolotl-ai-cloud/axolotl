# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Built-in quack ``autotune_profiles`` for SonicMoE's grouped GEMMs on B200.

Measured on ten expert-shape families (full fine-tuning and ``sonicmoe_lora_split``, EP=1 and
EP=8; kernels README, "Built-in profiles for SonicMoE on B200"): each grouped key sweeps at most
five candidates instead of 74-76 and stays within 2.3% of its best config. Every entry matches
only calls that pass ``cu_seqlens_m`` or ``cu_seqlens_k``, so dense quack GEMMs keep their own
search.
"""

import copy
import fnmatch
from typing import Any

_VARLEN_M = {"cu_seqlens_m.dim0": {"ge": 2}}
_VARLEN_K = {"cu_seqlens_k.dim0": {"ge": 2}}


def _config(
    tile_m, tile_n, cluster_m, cluster_n, persistent, tma_gather, swap_ab=False
):
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "is_dynamic_persistent": persistent,
        "use_tma_gather": tma_gather,
        "swap_ab": swap_ab,
    }


def _profile(kernel, match, *configs):
    return {
        "kernel": kernel,
        "backend": "quack",
        "gpu": "B200",
        "match": match,
        "configs": list(configs),
    }


# gemm_tuned entries go narrowest first: the first matching entry decides a call.
B200_SONICMOE_PROFILES: list[dict[str, Any]] = [
    # gate/up forward, with or without the LoRA addend: one config, not benchmarked
    _profile("gemm_gated_tuned", _VARLEN_M, _config(256, 512, 2, 1, True, True)),
    _profile(
        "gemm_dgated_tuned",
        _VARLEN_M,
        _config(256, 256, 2, 1, True, True),
        _config(128, 256, 2, 1, True, True),
        _config(256, 256, 2, 1, False, True),
    ),
    # expert weight grads
    _profile(
        "gemm_tuned",
        {**_VARLEN_K, "out.dim-1": {"gt": 128}, "out.dim-2": {"gt": 128}},
        _config(256, 512, 2, 1, True, True),
        _config(256, 256, 2, 1, True, True),
        _config(256, 256, 2, 1, False, False),
    ),
    # LoRA grads with r rows ([E, r, 2I])
    _profile(
        "gemm_tuned",
        {**_VARLEN_K, "out.dim-2": {"le": 128}},
        _config(256, 64, 2, 1, True, False, swap_ab=True),
        _config(128, 64, 1, 1, True, False, swap_ab=True),
    ),
    # LoRA grads with r columns ([E, H or I, r])
    _profile(
        "gemm_tuned",
        _VARLEN_K,
        _config(128, 64, 1, 1, True, False),
        _config(128, 64, 1, 1, False, False),
        _config(256, 64, 2, 1, True, False),
        _config(128, 64, 2, 1, True, False),
    ),
    # rank-r x P (N = r); the tma_gather ones only apply with A_idx
    _profile(
        "gemm_tuned",
        {**_VARLEN_M, "out.dim-1": {"le": 128}},
        _config(128, 64, 1, 1, True, False),
        _config(128, 64, 1, 1, False, False),
        _config(256, 64, 2, 1, True, False),
        _config(128, 64, 1, 1, True, True),
        _config(128, 64, 2, 1, True, True),
    ),
    # rank-r u Q (K = r)
    _profile(
        "gemm_tuned",
        {**_VARLEN_M, "B.dim-2": {"le": 128}},
        _config(256, 256, 2, 1, False, False),
        _config(128, 256, 1, 1, False, False),
    ),
    # down forward, gate/up dX, with or without the LoRA addend
    _profile(
        "gemm_tuned",
        _VARLEN_M,
        _config(256, 256, 2, 1, True, False),
        _config(256, 256, 2, 1, False, False),
        _config(256, 512, 2, 1, True, False),
        _config(128, 256, 1, 2, False, False),
    ),
]


def _names_kernel(pattern: str, kernel: str) -> bool:
    return fnmatch.fnmatchcase(kernel, pattern) or fnmatch.fnmatchcase(
        kernel, pattern.rpartition(".")[2]
    )


def default_quack_profiles(
    user_profiles: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """The built-in profiles for every kernel that ``user_profiles`` has no quack entry for.

    A kernel the config profiles at all keeps only the config's entries, so its calls that
    those entries do not match keep quack's full search.
    """
    taken = [
        p.get("kernel", "")
        for p in user_profiles or []
        if p.get("backend", "triton") == "quack"
    ]
    return [
        copy.deepcopy(profile)
        for profile in B200_SONICMOE_PROFILES
        if not any(_names_kernel(pattern, profile["kernel"]) for pattern in taken)
    ]
