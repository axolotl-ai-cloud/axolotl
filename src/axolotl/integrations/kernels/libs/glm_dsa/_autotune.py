"""Autotune helpers: portable shared-memory pruning.

Triton's autotune raises ``OutOfResources`` on a config whose tiles exceed the device's shared
memory (it does NOT silently skip it), so a config grid with big tiles crashes on a small-SMEM
GPU. ``smem_prune`` drops the over-budget configs up front by estimating each config's SMEM and
comparing to the *device's* real ``max_shared_mem`` (queried + memoized, not hardcoded) — so the
same grid runs on sm120 (~99 KB) and uses bigger tiles on Hopper/Blackwell-datacenter (~228 KB).
"""

from __future__ import annotations

import torch
import triton

_SMEM: dict[int, int] = {}


def max_smem(dev: int | None = None) -> int:
    dev = torch.cuda.current_device() if dev is None else dev
    if dev not in _SMEM:
        props = triton.runtime.driver.active.utils.get_device_properties(dev)
        _SMEM[dev] = int(props["max_shared_mem"])
    return _SMEM[dev]


def smem_prune(est_bytes, headroom: float = 0.95):
    """Build an ``early_config_prune`` callback. ``est_bytes(config_kwargs, num_stages, **constexprs)``
    returns a config's estimated SMEM in bytes; configs over ``headroom * device_limit`` are dropped
    (keeping the single smallest if all exceed, so autotune never gets an empty set)."""

    def prune(configs, named_args, **kwargs):
        limit = int(max_smem() * headroom)
        keep = [
            c for c in configs if est_bytes(c.kwargs, c.num_stages, **kwargs) <= limit
        ]
        if keep:
            return keep
        return sorted(
            configs, key=lambda c: est_bytes(c.kwargs, c.num_stages, **kwargs)
        )[:1]

    return prune
