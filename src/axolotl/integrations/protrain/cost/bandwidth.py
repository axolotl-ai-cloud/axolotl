"""Effective PCIe bandwidth model for the ProTrain cost estimators (§3.3).

When ``n_swap > 0`` activation-swap traffic (forward offload, backward
prefetch) competes with chunk prefetch/offload traffic on the same PCIe
link. ProTrain's cost model derates the prefetch bandwidth so the
runtime estimator does not under-predict backward time — but ONLY for
chunks whose prefetch window temporally overlaps the active swap
traffic. Per the paper:

    §3.3: "we estimate the swapping time, identify the affected chunks,
    and use the reduced bandwidth instead."

History
-------

The earliest implementation returned a single scalar derate per
direction and the runtime estimator applied it uniformly to every
chunk's prefetch — that over-penalised configurations because chunks
whose prefetch never overlaps a swap operation actually run at full
PCIe bandwidth.

A first refinement counted "affected chunks" (``n_affected_chunks``)
via the heuristic ``min(n_swap + 1, N_chunk - n_persist)`` and
distributed the derate across that count. This was an improvement but
still a count heuristic — it did not compute *which specific chunks*
have prefetch windows overlapping *which specific swap operations*; it
just charged the derate to the first ``n_affected`` non-persistent
chunks.

The current model is per-chunk timeline-overlap. For each chunk we
compute how many SWAP operations actually share PCIe with that chunk's
prefetch window (a function of the chunk's owning block, the runtime
scheduler's prefetch depth, and the per-block ``BlockStrategyMap``).
The derate factor scales with the overlap count, so a chunk whose
prefetch window never touches a SWAP block runs at full bandwidth even
when ``n_swap > 0`` elsewhere in the model.

API
---

- :func:`chunk_swap_overlap_count` — for a given chunk, count the
  SWAP-block operations that contend with its prefetch window.
- :func:`effective_bw_for_chunk` — per-chunk ``(eff_h2d, eff_d2h)``;
  full bandwidth when overlap is 0, ``raw / (1 + 0.5 * overlap)``
  otherwise. This is the primary path consumed by
  :func:`cost.runtime.estimate_runtime`.
- :func:`effective_bw` — legacy worst-case ``(eff_h2d, eff_d2h)`` pair.
  Retained for backward compatibility (the runtime ``Scheduler`` uses
  it as a single scalar to size its swap-stream pacing — see
  ``api.model_wrapper`` plumbing). NEW callers should prefer
  :func:`effective_bw_for_chunk`.
- :func:`n_affected_chunks` — legacy count heuristic. Retained for
  backward compatibility / external test consumers; the runtime no
  longer consumes it.

Paper references: §3.3 "bandwidth contention is modeled explicitly"; §A.1
swap-bandwidth identification.
"""

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


# Runtime scheduler's prefetch depth (``runtime/scheduler.py``):
#
# - ``pre_block_forward`` (line ~260): on entry to block ``b`` it kicks
#   off async gather for block ``b + 1``'s chunks on the prefetch
#   stream. That copy runs OVERLAPPED with block ``b``'s compute — so
#   block ``b + 1``'s chunk prefetch window IS block ``b``'s compute
#   window. ``prefetch_depth = 1``.
# - ``pre_block_backward`` (line ~411): symmetrically, on entry to
#   backward of block ``b`` it kicks off prefetch of block ``b - 1``'s
#   chunks. ``prefetch_depth = 1`` in backward as well.
#
# If a future scheduler refactor exposes a tunable lookahead, plumb it
# through here (e.g. via ``HardwareProfile`` or a ``CostConfig`` field)
# and pass it down to :func:`chunk_swap_overlap_count`. As of HEAD the
# lookahead is hardcoded to one block in both directions.
_DEFAULT_PREFETCH_DEPTH = 1


def effective_bw(cfg: CostConfig, hw: HardwareProfile) -> tuple[float, float]:
    """Legacy worst-case ``(effective_h2d_bps, effective_d2h_bps)`` pair.

    .. deprecated::
        Prefer :func:`effective_bw_for_chunk` for cost-model
        computations. This function is retained for the runtime
        :class:`~axolotl.integrations.protrain.runtime.scheduler.Scheduler`
        which consumes a single scalar pair to size its swap-stream
        pacing (the scheduler does not have a chunk-level view of
        which prefetch is active when, so the worst-case derate is
        the right thing to feed it).

    When ``cfg.n_swap == 0`` the raw PCIe bandwidths are returned
    unchanged (no swap traffic, nothing to contend with). When
    ``cfg.n_swap > 0`` the effective bandwidth is reduced by a factor
    ``1 / (1 + 0.5 * min(1, n_swap / max(1, gpu_count)))`` — the same
    arithmetic the per-chunk path applies when ``overlap_count == 1``,
    matching the paper's qualitative claim that "unlimited" swap
    degrades prefetch throughput by roughly a third.

    Parameters
    ----------
    cfg:
        The candidate knob configuration being costed.
    hw:
        Static hardware description; only ``pcie_h2d_bps``,
        ``pcie_d2h_bps``, and ``gpu_count`` are consulted.

    Returns
    -------
    tuple[float, float]
        Worst-case effective H2D and D2H bandwidths in bytes / second.
    """
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
    """Legacy count of non-persistent chunks overlapping swap traffic.

    .. deprecated::
        The runtime estimator now uses
        :func:`chunk_swap_overlap_count` per-chunk. This count
        heuristic is retained as a coarse summary for external test
        consumers and back-compat with pre-timeline callers.

    Formula (unchanged from the count-heuristic implementation):
    ``n_affected = min(n_swap + 1, N_chunk - n_persist)``.

    Returns 0 when ``n_swap <= 0`` (no swap traffic, no contention).
    """
    if cfg.n_swap <= 0:
        return 0
    # Mirror the rest of the cost model: the augmented persistent set
    # ``{0..n_persist-1} union mandatory_persistent`` defines the GPU-resident
    # population, so the count of chunks subject to swap-induced PCIe
    # contention is ``N_chunk - len(augmented)``.
    n_persist_eff = len(layout.effective_persistent_ids(cfg.n_persist))
    n_nonpersist = max(0, layout.N_chunk - n_persist_eff)
    return min(cfg.n_swap + 1, n_nonpersist)


# Per-layout reverse map cache. ``chunk_swap_overlap_count`` is called
# once per chunk inside the searcher's hot loop, so the original
# implementation that rescanned ``layout.block_to_chunks`` for every
# lookup was O(N_block) per chunk. We memoize the chunk -> owner-blocks
# map keyed on ``id(layout)``; ChunkLayout is a frozen dataclass so the
# id is stable for its lifetime.
#
# ``id()`` values are recycled by CPython once a layout is GC'd, so a
# new ``ChunkLayout`` could otherwise pick up a stale cache entry from
# a freed object. ``ChunkLayout`` carries ``dict`` fields and is not
# hashable, which rules out ``WeakKeyDictionary``. Instead we register
# a ``weakref.finalize`` callback at insertion time: when the layout
# is GC'd, the finalizer evicts its cache entry by id, eliminating the
# stale-key window.
_CHUNK_TO_OWNERS_CACHE: dict[int, dict[int, list[BlockId]]] = {}


def _evict_chunk_owners_cache(layout_key: int) -> None:
    """Drop the cache entry keyed on ``layout_key`` (called from a finalizer)."""
    _CHUNK_TO_OWNERS_CACHE.pop(layout_key, None)


def _block_of_chunk(chunk_id: ChunkId, layout: ChunkLayout) -> list[BlockId]:
    """Return the block(s) owning ``chunk_id``.

    ``layout.block_to_chunks`` maps block -> tuple[ChunkId]. For a
    typical transformer layout each chunk belongs to exactly one
    block, but the data structure supports many-to-many (intra-chunk
    block-shared paths). We return the list of owning blocks so the
    overlap calculation can union prefetch-source windows across all
    owners — defensive against fixtures and real layouts where a
    chunk straddles a block boundary.
    """
    layout_key = id(layout)
    owners_map = _CHUNK_TO_OWNERS_CACHE.get(layout_key)
    if owners_map is None:
        owners_map = {}
        for bid, cids in layout.block_to_chunks.items():
            for cid in cids:
                owners_map.setdefault(int(cid), []).append(bid)
        _CHUNK_TO_OWNERS_CACHE[layout_key] = owners_map
        # Auto-evict the cache entry when the layout is garbage
        # collected so a recycled ``id()`` cannot resurface stale
        # data on a future, unrelated layout. ``finalize`` holds only
        # a weak reference to the layout, so registering the callback
        # does not extend its lifetime.
        weakref.finalize(layout, _evict_chunk_owners_cache, layout_key)
    return list(owners_map.get(int(chunk_id), ()))


def chunk_swap_overlap_count(
    chunk_id: ChunkId,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    prefetch_depth: int = _DEFAULT_PREFETCH_DEPTH,
    direction: str = "fwd",
) -> int:
    """How many SWAP operations overlap ``chunk_id``'s prefetch window.

    A chunk belonging to block ``b`` is prefetched while block
    ``b - prefetch_depth`` (forward) or ``b + prefetch_depth``
    (backward) is computing — that is the prefetch SOURCE window. If
    any block within that source window is a SWAP block, its D2H
    (forward) / H2D (backward) activation traffic competes with this
    chunk's prefetch on the shared PCIe link.

    Concretely, with ``prefetch_depth = 1``:

    - **Forward**: chunk for block ``b`` is gathered onto the prefetch
      stream during the compute of block ``b - 1``. If block ``b - 1``
      is SWAP, its activation D2H is in flight on the swap stream
      during the same window → 1 overlap.
    - **Backward**: chunk for block ``b`` is gathered during the
      backward compute of block ``b + 1``. If block ``b + 1`` is SWAP,
      its activation H2D is in flight → 1 overlap.

    For chunks owned by multiple blocks (rare, but supported by the
    ``ChunkLayout`` schema) the per-owner source windows are unioned
    and SWAP blocks in the union are counted once each. ``ChunkId(0)``
    has no source window in forward (no preceding block to prefetch
    from); we return 0 in that case — the chunk is gathered
    synchronously by the first-block warm-up in
    ``Scheduler.pre_block_forward`` and pays no contention from the
    one prefetch-depth lookahead path.

    Parameters
    ----------
    chunk_id:
        Chunk whose prefetch window we're costing.
    layout:
        Chunk layout, consulted for ``block_to_chunks``.
    block_map:
        Per-block mode assignment; SWAP blocks contribute to overlap.
    prefetch_depth:
        Number of compute blocks the prefetch trails the consumer
        block. Defaults to 1, matching the runtime scheduler's
        ``pre_block_forward`` / ``pre_block_backward`` lookahead. If a
        scheduler refactor introduces a tunable depth, pass it through
        here.
    direction:
        ``"fwd"`` or ``"bwd"``. Determines whether the source window is
        on the lower-indexed (forward) or higher-indexed (backward)
        side of the consumer block.

    Returns
    -------
    int
        Number of SWAP blocks whose forward D2H (forward direction) or
        backward H2D (backward direction) traffic overlaps this
        chunk's prefetch. Range ``[0, prefetch_depth * len(owners)]``.
    """
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"direction must be 'fwd' or 'bwd', got {direction!r}")
    if prefetch_depth < 1:
        raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth!r}")

    owners = _block_of_chunk(chunk_id, layout)
    if not owners:
        return 0

    # Source-block index range. ``layout.block_to_chunks`` may be sparse
    # in pathological fixtures; we don't iterate it here — we just
    # consult ``block_map.get(...)`` which returns ``BlockMode.NONE`` by
    # convention for missing blocks (the cost-model invariant; see
    # ``layout_rules.assign_modes`` which always populates every block).
    source_blocks: set[BlockId] = set()
    for owner in owners:
        b = int(owner)
        if direction == "fwd":
            # Forward: prefetch for block b runs during compute of
            # blocks [b - prefetch_depth, b - 1]. Negative indices
            # don't exist (first-block warm-up handles them) — clamp
            # to 0.
            for offset in range(1, prefetch_depth + 1):
                src = b - offset
                if src >= 0:
                    source_blocks.add(BlockId(src))
        else:
            # Backward: prefetch for block b runs during backward of
            # blocks [b + 1, b + prefetch_depth]. There is no upper
            # clamp — backward of block N_block-1 has no successor and
            # contributes no overlap.
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
    """Per-chunk ``(eff_h2d_bps, eff_d2h_bps)`` under timeline-aware contention.

    Returns the FULL PCIe bandwidth pair when the chunk's prefetch
    window overlaps no SWAP operations (overlap_count = 0). Otherwise
    derates by ``1 / (1 + 0.5 * overlap_count)``, matching the
    qualitative paper bound (one swap source ≈ 1.5x denominator,
    matching the legacy worst-case derate).

    Parameters
    ----------
    chunk_id:
        Chunk whose prefetch is being costed.
    cfg:
        Candidate knob configuration. Consulted for ``n_swap``: when
        ``n_swap == 0`` we short-circuit to full bandwidth without
        traversing the layout (cheap fast path for the no-swap case).
    hw:
        Static hardware description; ``pcie_h2d_bps`` and
        ``pcie_d2h_bps`` are the raw bandwidths to derate.
    layout, block_map:
        Forwarded to :func:`chunk_swap_overlap_count`.
    direction:
        ``"fwd"`` or ``"bwd"``.
    prefetch_depth:
        Forwarded to :func:`chunk_swap_overlap_count`.

    Returns
    -------
    tuple[float, float]
        ``(eff_h2d_bps, eff_d2h_bps)`` for this chunk.
    """
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
