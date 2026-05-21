"""Effective PCIe bandwidth model: per-chunk timeline overlap with SWAP traffic."""

from __future__ import annotations

import weakref

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# Scheduler's single-block lookahead in both fwd and bwd.
_DEFAULT_PREFETCH_DEPTH = 1


def effective_bw(cfg: CostConfig, hw: HardwareProfile) -> tuple[float, float]:
    """Legacy worst-case (eff_h2d_bps, eff_d2h_bps); prefer effective_bw_for_chunk."""
    gpu_count = max(1, hw.gpu_count)
    if cfg.n_swap <= 0:
        return hw.pcie_h2d_bps, hw.pcie_d2h_bps

    contention = 0.5 * min(1.0, cfg.n_swap / gpu_count)
    denom = 1.0 + contention
    eff_h2d = hw.pcie_h2d_bps / denom
    eff_d2h = hw.pcie_d2h_bps / denom
    LOG.debug(
        "effective_bw (legacy): n_swap=%d gpu_count=%d derate=%.3f h2d=%.2e d2h=%.2e",
        cfg.n_swap,
        gpu_count,
        denom,
        eff_h2d,
        eff_d2h,
    )
    return eff_h2d, eff_d2h


def n_affected_chunks(cfg: CostConfig, layout: ChunkLayout) -> int:
    """Legacy count heuristic: min(n_swap+1, N_chunk - n_persist); 0 when n_swap<=0."""
    if cfg.n_swap <= 0:
        return 0
    n_persist_eff = len(layout.effective_persistent_ids(cfg.n_persist))
    n_nonpersist = max(0, layout.N_chunk - n_persist_eff)
    return min(cfg.n_swap + 1, n_nonpersist)


# Per-layout chunk→owners cache; weakref.finalize evicts on layout GC to avoid id-recycle stale hits.
_CHUNK_TO_OWNERS_CACHE: dict[int, dict[int, list[BlockId]]] = {}


def _evict_chunk_owners_cache(layout_key: int) -> None:
    """Drop the cache entry keyed on ``layout_key`` (called from a finalizer)."""
    _CHUNK_TO_OWNERS_CACHE.pop(layout_key, None)


def _block_of_chunk(chunk_id: ChunkId, layout: ChunkLayout) -> list[BlockId]:
    """Return block(s) owning chunk_id; supports many-to-many for boundary-straddling chunks."""
    layout_key = id(layout)
    owners_map = _CHUNK_TO_OWNERS_CACHE.get(layout_key)
    if owners_map is None:
        owners_map = {}
        for bid, cids in layout.block_to_chunks.items():
            for cid in cids:
                owners_map.setdefault(int(cid), []).append(bid)
        _CHUNK_TO_OWNERS_CACHE[layout_key] = owners_map
        # weakref.finalize doesn't extend layout lifetime; evicts on GC.
        weakref.finalize(layout, _evict_chunk_owners_cache, layout_key)
    return list(owners_map.get(int(chunk_id), ()))


def chunk_swap_overlap_count(
    chunk_id: ChunkId,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    prefetch_depth: int = _DEFAULT_PREFETCH_DEPTH,
    direction: str = "fwd",
) -> int:
    """Count SWAP-block ops overlapping chunk_id's prefetch window (fwd or bwd)."""
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"direction must be 'fwd' or 'bwd', got {direction!r}")
    if prefetch_depth < 1:
        raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth!r}")

    owners = _block_of_chunk(chunk_id, layout)
    if not owners:
        return 0

    source_blocks: set[BlockId] = set()
    for owner in owners:
        b = int(owner)
        if direction == "fwd":
            # Source = [b-prefetch_depth, b-1]; clamp negative to 0.
            for offset in range(1, prefetch_depth + 1):
                src = b - offset
                if src >= 0:
                    source_blocks.add(BlockId(src))
        else:
            # Source = [b+1, b+prefetch_depth]; no upper clamp.
            for offset in range(1, prefetch_depth + 1):
                src = b + offset
                source_blocks.add(BlockId(src))

    overlap = 0
    for src in source_blocks:
        if block_map.get(src, BlockMode.NONE) is BlockMode.SWAP:
            overlap += 1
    return overlap


def effective_bw_for_chunk(
    chunk_id: ChunkId,
    cfg: CostConfig,
    hw: HardwareProfile,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    direction: str = "fwd",
    prefetch_depth: int = _DEFAULT_PREFETCH_DEPTH,
) -> tuple[float, float]:
    """Per-chunk eff bandwidth: raw when no SWAP overlap, else raw/(1 + 0.5*overlap_count)."""
    # Validate kwargs BEFORE no-swap fast path for consistent error semantics.
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"direction must be 'fwd' or 'bwd', got {direction!r}")
    if prefetch_depth < 1:
        raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth!r}")

    # Fast path: no swap blocks anywhere → no chunk can overlap swap.
    if cfg.n_swap <= 0:
        return hw.pcie_h2d_bps, hw.pcie_d2h_bps

    overlap = chunk_swap_overlap_count(
        chunk_id, layout, block_map, prefetch_depth=prefetch_depth, direction=direction
    )
    if overlap <= 0:
        return hw.pcie_h2d_bps, hw.pcie_d2h_bps

    denom = 1.0 + 0.5 * overlap
    return hw.pcie_h2d_bps / denom, hw.pcie_d2h_bps / denom


__all__ = [
    "chunk_swap_overlap_count",
    "effective_bw",
    "effective_bw_for_chunk",
    "n_affected_chunks",
]
