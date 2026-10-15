# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Narrow Triton autotune searches to the neighborhood of an already measured winner.

The ScatterMoE LoRA kernels key their autotune on the routed-row bucket, so every new bucket
re-benchmarks the full tile grid (over a hundred configs for the NVFP4 variant), one rank at a
time while its expert-parallel peers wait. Winners barely move between buckets of the same
``(N, K)``: a real-width profiling run picked the same tile for every bucket it saw. So once an
autotuner holds a winner for a key that differs only in the bucketed arguments, a new key only
benchmarks configs within ``distance`` parameter changes of that winner (distance 1 = vary one
of BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages; 2 = pairs).
"""

from typing import Any

from axolotl.integrations.kernels.autotune_cache import _autotuners
from axolotl.integrations.kernels.autotune_collector import _config_to_dict
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# key arguments that carry a size bucket rather than a shape: neighbors may differ on them
BUCKET_KEYS = ("M_BUCKET", "B_BIN", "M")


def _config_fields(config: Any) -> dict[str, Any]:
    return {k: v for k, v in _config_to_dict(config).items() if v is not None}


def _distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    names = set(a) | set(b)
    return sum(1 for n in names if a.get(n) != b.get(n))


def nearest_winner(autotuner: Any, key: tuple) -> Any | None:
    """A cached winner whose key matches ``key`` on every non-bucket argument, closest bucket first."""
    names = list(getattr(autotuner, "keys", None) or [])
    if not names or not autotuner.cache:
        return None
    bucket_idx = [i for i, n in enumerate(names) if n in BUCKET_KEYS]
    if not bucket_idx:
        return None
    fixed = [i for i in range(len(key)) if i not in bucket_idx]
    best, best_gap = None, None
    for cached_key, config in autotuner.cache.items():
        if len(cached_key) < len(key) or any(cached_key[i] != key[i] for i in fixed):
            continue
        gap = sum(
            abs(float(cached_key[i]) - float(key[i]))
            for i in bucket_idx
            if isinstance(cached_key[i], (int, float))
            and isinstance(key[i], (int, float))
        )
        if best_gap is None or gap < best_gap:
            best, best_gap = config, gap
    return best


def neighborhood_configs(
    autotuner: Any, configs: list, winner: Any, distance: int
) -> list:
    center = _config_fields(winner)
    near = [c for c in configs if _distance(_config_fields(c), center) <= distance]
    # the winner may be a config rebuilt from the cache file rather than a member of
    # the kernel's list; it was measured for this kernel, so it is always a candidate
    if winner not in near:
        near.insert(0, winner)
    return near


def install_neighborhood_search(distance: int = 1) -> int:
    """Wrap ``prune_configs`` on every loaded autotuner. Returns how many were wrapped."""
    if distance < 1:
        raise ValueError("autotune_neighborhood must be at least 1")
    installed = 0
    for _modname, _module, autotuner in _autotuners():
        if getattr(autotuner, "_axolotl_neighborhood", None) is not None:
            autotuner._axolotl_neighborhood = distance
            continue
        if not getattr(autotuner, "keys", None):
            continue
        orig_prune = autotuner.prune_configs
        autotuner._axolotl_neighborhood = distance

        def prune_configs(kwargs, _tuner=autotuner, _orig=orig_prune):
            configs = _orig(kwargs)
            if len(configs) <= 1:
                return configs
            args = {**(getattr(_tuner, "nargs", None) or {}), **kwargs}
            try:
                key = tuple(args[name] for name in _tuner.keys if name in args)
            except Exception:  # pylint: disable=broad-exception-caught
                return configs
            # Triton appends the dtype signature; cached keys carry it, the arg-only key does not,
            # so compare on the declared-argument prefix.
            winner = nearest_winner(_tuner, key)
            if winner is None:
                return configs
            near = neighborhood_configs(
                _tuner, configs, winner, _tuner._axolotl_neighborhood
            )
            return near or configs

        autotuner.prune_configs = prune_configs
        installed += 1
    if installed:
        LOG.info(
            "Autotune neighborhood search (distance %d) attached to %d kernel(s)",
            distance,
            installed,
        )
    return installed
