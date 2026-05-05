"""Exhaustive 5-knob search for ProTrain (§3.3, Option B §4.3).

Algorithm:

1. Derive ``Bounds`` from ``(trace, layout)``.
2. Enumerate ``(n_persist, n_buffer, n_swap, n_checkpoint, n_offload)``
   within bounds, subject to:

   - ``n_persist + n_buffer <= N_chunk``
   - ``n_swap + n_checkpoint + n_offload <= N_block``
   - ``n_swap <= min(N_block - n_checkpoint - n_offload, N_interval)``

3. For each candidate, compute ``block_map = assign_modes(...)``.
4. Evaluate ``estimate_peak``; drop candidates above ``capacity_bytes``.
5. Drop runtime-inadmissible candidates: any block whose parameter
   chunks are not all persistent must use ``CKPT`` or ``OFFLOAD``,
   because the current runtime releases non-persistent chunk storage
   after forward and relies either on checkpoint recomputation
   (``CKPT``) or on the OFFLOAD saved-tensors-hook re-bind path
   (``OFFLOAD``) to make activations available again for backward.
   See ``block_map_runtime_admissible`` for the precise predicate.
6. If ``cpu_capacity_bytes`` is not None, evaluate
   ``estimate_cpu_footprint``; drop candidates above the host-RAM gate.
7. Among survivors, evaluate ``estimate_runtime`` and pick argmin.
8. Raise ``RuntimeError`` if no candidate fits — the message
   distinguishes GPU-pressure failure (no cfg cleared the GPU gate)
   from CPU-pressure failure (some cleared GPU but all busted CPU).

The search space is tiny (~10^4 at most on realistic models even with
the added ``n_offload`` axis) — no pruning cleverness is needed for
correctness. We do sort candidates by a cheap static peak estimate so
early OOMs filter out large chunks of the space without the full
op-walk.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Iterator

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost.memory import (  # noqa: F401 - re-exported for test back-compat
    estimate_cpu_footprint,
    estimate_peak,
)
from axolotl.integrations.protrain.cost.runtime import estimate_runtime
from axolotl.integrations.protrain.search.knobs import derive_bounds
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    Bounds,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
    SearchResult,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def min_n_buffer_for(layout: ChunkLayout, n_persist: int) -> int:
    """Minimum n_buffer the scheduler needs at this n_persist.

    The scheduler's lookahead prefetch (runtime/scheduler.py::pre_block_forward)
    holds the current block's chunks resident while simultaneously prefetching
    the next block's chunks. For any non-persistent chunk to be reachable via
    the pool, the pool must be sized for the worst-case union across adjacent
    block pairs. Persistent chunks (the first ``n_persist``) bypass the pool,
    so we only count non-persistent contributions.

    Returns 0 when every chunk is persistent (``n_persist >= N_chunk``).
    """
    if n_persist >= layout.N_chunk:
        return 0
    persistent: set[ChunkId] = {ChunkId(i) for i in range(n_persist)}
    block_ids = sorted(layout.block_to_chunks.keys())
    if not block_ids:
        # Sparse/degenerate layout: ``n_persist < N_chunk`` above means at
        # least one chunk is non-persistent, but block_to_chunks doesn't
        # surface which block owns it. The pool allocator still needs one
        # slot to materialize that chunk, so honour the same ``max(1, …)``
        # invariant the dense branch enforces below.
        return 1
    need = 0
    for i, bid in enumerate(block_ids):
        cur_np = [c for c in layout.block_to_chunks.get(bid, ()) if c not in persistent]
        nxt_np: list[ChunkId] = []
        if i + 1 < len(block_ids):
            nxt_np = [
                c
                for c in layout.block_to_chunks.get(block_ids[i + 1], ())
                if c not in persistent
            ]
        need = max(need, len({*cur_np, *nxt_np}))
    # Every pool allocator path requires at least 1 buffer when any
    # non-persistent chunk exists, even if block_to_chunks is sparse.
    return max(1, need)


def block_map_runtime_admissible(
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    n_persist: int,
) -> bool:
    """Return True iff the block strategy is safe for current chunk offload.

    Four-mode admissibility (post-Option B; see
    ``BLOCK_MODE_OFFLOAD_DESIGN.md`` §3.5):

    * ``CKPT`` — always admissible. The recompute path re-binds storage by
      replaying the wrapped forward inside ``torch.utils.checkpoint``; the
      scheduler re-gathers the block's chunks immediately before recompute.
    * ``OFFLOAD`` — always admissible. The wrapper installs a
      saved-tensors-hook that records metadata only at pack time and
      re-gathers the chunk at unpack time, so post-forward chunk release is
      safe even with non-persistent params.
    * ``NONE`` and ``SWAP`` — admissible iff every chunk owned by the
      block is in the persistent set. The forward scheduler releases
      non-persistent chunk storage after the block runs, and PyTorch's
      saved tensors for a normal NONE/SWAP block are not a safe
      persistence mechanism once ``param.data`` is rebound to the empty
      sentinel. NONE/SWAP on a block with any non-persistent chunk
      remains inadmissible.

    Fully persistent blocks may use NONE/SWAP because their parameter
    storage is never nulled or recycled.
    """
    persistent = {ChunkId(i) for i in range(max(0, int(n_persist)))}
    for bid, chunks in layout.block_to_chunks.items():
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.CKPT or mode is BlockMode.OFFLOAD:
            # CKPT recomputes; OFFLOAD's saved-tensors-hook re-binds
            # storage at backward — both safe regardless of persistence.
            continue
        if any(ChunkId(int(cid)) not in persistent for cid in chunks):
            return False
    return True


def _iter_candidates(bounds: Bounds) -> Iterator[CostConfig]:
    """Enumerate feasible ``CostConfig`` tuples within ``bounds``.

    Five axes (Option B §4.3): ``n_checkpoint``, ``n_offload``,
    ``n_swap``, ``n_persist``, ``n_buffer``. ``n_offload`` lives in
    the outer-loop neighbourhood of ``n_ckpt`` because the two trade
    against each other on the backward wall (Option B §4.2). Search
    space grows by ~``N_block`` (~17K -> ~440K candidates on a
    Llama-3B-class model with ``N_block=26``), still well under the
    second-budget for closed-form per-candidate evaluation.
    """
    n_chunk = bounds.N_chunk
    n_block = bounds.N_block
    n_interval = bounds.N_interval

    for n_ckpt in range(0, n_block + 1):
        for n_offload in range(0, n_block - n_ckpt + 1):
            # n_swap bounded by (a) blocks remaining after
            # ckpt+offload, (b) N_interval.
            max_swap = min(n_block - n_ckpt - n_offload, n_interval)
            for n_swap in range(0, max_swap + 1):
                for n_persist in range(0, n_chunk + 1):
                    # n_buffer fills the remainder of chunk budget.
                    max_buffer = n_chunk - n_persist
                    for n_buffer in range(0, max_buffer + 1):
                        yield CostConfig(
                            n_persist=n_persist,
                            n_buffer=n_buffer,
                            n_swap=n_swap,
                            n_checkpoint=n_ckpt,
                            n_offload=n_offload,
                        )


def _block_map_peak_contribution(
    block_map: BlockStrategyMap,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    *,
    forward_ops_by_block: dict[BlockId, list[int]] | None = None,
    tree_index_map: dict[BlockId, int] | None = None,
    n_persist: int | None = None,
) -> int:
    """Compute the block-map-dependent part of the raw peak.

    Matches the op-walk inside :func:`estimate_peak` but returns only
    the terms that do not depend on ``(n_persist, n_buffer)``:

        F(block_map) = max over forward ops i of
            (live_none_at(i) + ckpt_extra_at(i) + offload_extra_at(i)
             + cross_attn_at(i) + intra[i] + inter[i])

    The returned value is the pre-alpha raw contribution; the caller
    multiplies the full ``model_state_present + F`` sum by
    ``ALPHA_FRAGMENTATION`` and ``int()``-casts to match
    ``estimate_peak`` exactly.

    ``forward_ops_by_block`` and ``tree_index_map`` depend only on
    ``trace`` (not ``block_map``); when called inside the searcher's
    hot loop callers should compute them once and pass them in to
    skip the per-iteration rebuild.

    The OFFLOAD bump term (``offload_extra_at``) lands at the LAST
    forward op of each OFFLOAD block (Option B §4.1) and contributes
    ``layout.S_chunk`` (the buffer-pool chunk gather only —
    activations are already counted in ``live_none`` because OFFLOAD
    retains them like NONE). The ``layout`` parameter is required to
    provide ``S_chunk``.

    ``n_persist``: when provided, the OFFLOAD bump is suppressed for
    OFFLOAD blocks whose chunks are ALL in the persistent set
    (``chunk_id < n_persist``). Rationale: ``ChunkManager.gather`` is
    a no-op for persistent chunks (see ``chunk/manager.py::gather``
    "Persistent chunks: no-op — they were never offloaded"), so the
    backward-window chunk-gather residency that the bump models does
    not occur when the block's chunks are already GPU-resident. When
    ``n_persist`` is ``None`` (legacy callers — ``estimate_peak``'s
    full op-walk path), every OFFLOAD block contributes the bump.
    The searcher's hot loop varies ``n_persist`` independently of
    ``block_map`` and so MUST pass this argument to avoid over-stating
    the peak for high-``n_persist`` OFFLOAD configs (which would
    spuriously prune feasible candidates via the ``max_sum`` ceiling
    derived from ``f_bm``).

    Cross-attention term mirrors ``estimate_peak``'s Fix-3 enc-dec
    accounting — see the docstring of that function. For single-tree
    causal-LM traces the term is 0 and this matches the legacy F_bm.
    """
    from axolotl.integrations.protrain.cost.memory import (
        block_tree_index_map,
        cross_attn_persist_bytes,
        op_cross_attn_surcharge,
    )

    if forward_ops_by_block is None:
        forward_ops_by_block = defaultdict(list)
        for i, op in enumerate(trace.op_order):
            if op.is_forward and op.block_id is not None:
                forward_ops_by_block[op.block_id].append(i)

    # Identify CKPT bump ops (first forward op of each CKPT block) and
    # OFFLOAD bump ops (last forward op of each OFFLOAD block — closest
    # forward index to that block's first backward op). When
    # ``n_persist`` is provided, an OFFLOAD block whose chunks are ALL
    # within the persistent set contributes NO bump — the runtime
    # ``ChunkManager.gather`` short-circuits for persistent chunks so
    # no backward-window chunk-buffer materialization happens.
    ckpt_bump_op: dict[int, int] = {}
    offload_bump_op: dict[int, int] = {}
    persistent_chunks: set[ChunkId] | None = None
    if n_persist is not None:
        persistent_chunks = {ChunkId(i) for i in range(max(0, int(n_persist)))}
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)
        elif mode is BlockMode.OFFLOAD:
            if persistent_chunks is not None:
                chunks = layout.block_to_chunks.get(block_id, ())
                # All-persistent OFFLOAD block: gather is a no-op, so
                # no backward chunk-buffer materialization. ``chunks``
                # may be empty for sparse/degenerate layouts; treat
                # that as "no bump" since there's no chunk to gather.
                if not chunks or all(
                    ChunkId(int(cid)) in persistent_chunks for cid in chunks
                ):
                    continue
            offload_bump_op[op_idxs[-1]] = int(block_id)

    # Cumulative NONE / OFFLOAD activation bytes at each forward-op index.
    # OFFLOAD retains activations on GPU symmetrically to NONE; the
    # additional chunk gather bump fires at the per-block backward window
    # via ``offload_bump_op`` and is added separately below.
    block_first_op = {bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops}
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])
    cumulative_none: list[tuple[int, int]] = []  # (first_op_idx, cumulative)
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            running += trace.activation_sizes.get(bid, 0)
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    if tree_index_map is None:
        tree_index_map = block_tree_index_map(trace)
    cross_attn_bytes = cross_attn_persist_bytes(trace, block_map, tree_index_map)

    s_chunk = layout.S_chunk
    best = 0
    have_any_forward = False
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        have_any_forward = True
        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)
        ckpt_extra = 0
        if i in ckpt_bump_op:
            ckpt_extra = trace.activation_sizes.get(BlockId(ckpt_bump_op[i]), 0)
        offload_extra = 0
        if i in offload_bump_op:
            offload_extra = s_chunk
        op_cross_attn = op_cross_attn_surcharge(op, cross_attn_bytes, tree_index_map)
        candidate = (
            live_none + ckpt_extra + offload_extra + op_cross_attn + intra + inter
        )
        if candidate > best:
            best = candidate

    if not have_any_forward:
        # Degenerate trace: fall back to the NONE/OFFLOAD retained-
        # activation total so the caller's peak is at least
        # ``model_state_present + retained``. (OFFLOAD retains
        # activations like NONE — the chunk-gather bump term would
        # only fire during the op-walk if forward ops were present.)
        total_none = 0
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            mode = block_map.get(bid, BlockMode.NONE)
            if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
                total_none += act_sz
        return total_none

    return best


def search(
    trace: ProfilerTrace,
    layout: ChunkLayout,
    capacity_bytes: int,
    hw: HardwareProfile,
    cpu_capacity_bytes: int | None = None,
) -> SearchResult:
    """Return the minimum-runtime ``SearchResult`` fitting under
    ``capacity_bytes`` (and ``cpu_capacity_bytes`` when provided).

    Parameters
    ----------
    trace, layout, hw:
        See module docstring.
    capacity_bytes:
        GPU per-rank memory budget. Configs whose predicted peak
        exceeds this are dropped before runtime evaluation.
    cpu_capacity_bytes:
        Optional per-rank pinned CPU RAM budget. When provided,
        configs whose ``estimate_cpu_footprint`` exceeds this are
        also dropped — the searcher then guarantees its pick fits
        BOTH the GPU and CPU envelopes. ``None`` (the default)
        preserves the pre-CPU-filter behaviour for backward
        compatibility.

    Raises
    ------
    RuntimeError
        If no candidate clears both the GPU capacity gate and the
        optional CPU capacity gate. The message distinguishes the two
        failure modes so callers can tell whether to scale up GPU
        memory or host RAM.

    Notes
    -----
    Correctness is equivalent to the naive 5-loop enumeration over
    ``(n_persist, n_buffer, n_swap, n_ckpt, n_offload)`` that calls
    ``estimate_peak`` and ``estimate_runtime`` inside the inner
    (n_persist, n_buffer) iteration. We exploit two structural
    invariants to avoid quadratic op-walks across the full search
    space:

    1. ``estimate_peak``'s raw peak decomposes as
       ``model_state_present(cfg, layout, trace) + F(block_map)``,
       where ``model_state_present`` is the persistent + buffer-pool
       residency from ``cost/memory.py::model_state_present_bytes``
       (full Adam state per persistent chunk under fp16+full-FT,
       fp16-only under LoRA-with-frozen-base; see Eq. 11 derivation
       in ``estimate_peak``). The block-map-dependent term ``F`` is
       independent of ``(n_persist, n_buffer)`` so we compute it once
       per ``(n_swap, n_ckpt, n_offload)`` triple
       (O(N_swap*N_ckpt*N_offload*N_op)).
    2. ``estimate_runtime`` is a closed-form function of the config,
       evaluated only for configs that already clear the capacity
       gate — keeping the inner loop purely arithmetic.

    For a 7B-class model this cuts the search from ~50 billion op-walk
    iterations down to ~1 million, without changing the selected
    ``(cfg, block_map)``.
    """
    bounds = derive_bounds(trace, layout)

    # Under ZeRO-3 sharding (``hw.zero3_shard=True``) each rank holds
    # only ``chunk_bytes / world_size`` per non-persistent chunk on
    # CPU, so the CPU-pressure constraint that would otherwise shrink
    # viable ``n_buffer`` ceilings goes away. We therefore let
    # ``n_buffer`` roam up to its natural upper bound of
    # ``N_chunk - n_persist`` in both modes — the search's GPU-capacity
    # gate (``predicted_peak > capacity_bytes``) is the only
    # feasibility filter, and it is sharding-agnostic because the
    # gather materializes the full chunk on GPU regardless. See
    # ``cost/memory.py::estimate_cpu_footprint`` for the per-rank CPU
    # accounting that would feed a tighter CPU-budget filter if one
    # is added downstream.
    _ = hw.zero3_shard  # noqa: F841 — explicit acknowledgement

    n_total = 0
    n_feasible = 0
    n_gpu_feasible = 0  # cleared GPU gate (used to disambiguate failure mode)
    n_cpu_rejected = 0  # cleared GPU gate but failed CPU gate
    # cleared GPU+CPU gates but estimate_runtime returned non-finite
    n_runtime_rejected = 0
    best_iter_s: float = float("inf")
    best_cfg: CostConfig | None = None
    best_block_map: BlockStrategyMap | None = None
    best_peak: int = 0

    # Pre-compute block-map-dependent terms once per (n_swap, n_ckpt).
    # ``F(block_map)`` is the raw-peak contribution excluding the
    # ``(n_persist + n_buffer) * S_chunk`` term, pre-alpha.
    from axolotl.integrations.protrain.cost.memory import (
        ALPHA_FRAGMENTATION,
        block_tree_index_map,
        hot_iter_peak_cap,
        model_state_present_bytes,
    )

    alpha = ALPHA_FRAGMENTATION
    s_chunk = layout.S_chunk

    # Hoist trace-only maps out of the (n_swap, n_ckpt) hot loop —
    # both depend on ``trace`` only, not ``block_map``.
    forward_ops_by_block: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if op.is_forward and op.block_id is not None:
            forward_ops_by_block[op.block_id].append(i)
    tree_index_map = block_tree_index_map(trace)

    for n_ckpt in range(0, bounds.N_block + 1):
        # Option B §4.3: outer loop over n_offload — added as a sibling
        # axis to n_ckpt because the two trade against each other on the
        # backward wall (Option B §4.2). Search space grows ~N_block-fold but
        # the per-candidate work is closed-form so it stays sub-second on
        # realistic Llama-3B/7B-class models.
        for n_offload in range(0, bounds.N_block - n_ckpt + 1):
            max_swap = min(bounds.N_block - n_ckpt - n_offload, bounds.N_interval)
            for n_swap in range(0, max_swap + 1):
                block_map = assign_modes(
                    n_swap, n_ckpt, bounds.N_block, n_offload=n_offload
                )

                # For a fixed (n_ckpt, n_swap) sweep n_persist. The optimal
                # n_buffer at each n_persist is the maximum feasible value
                # in [0, N_chunk - n_persist]: ``estimate_runtime``'s
                # n_buffer dependence enters only through ``n_cached =
                # min(n_buffer, n_nonpersist)`` inside the backward
                # communication term, and
                # ``max(compute, comm_cached) <= max(compute, comm_uncached)``
                # because cached chunks skip the re-gather. So moving a
                # chunk from uncached to cached never increases ``t_iter``;
                # the argmin is reached by maximising n_buffer within
                # capacity. That collapses the inner (n_persist, n_buffer)
                # loop from O(N_chunk^2) to O(N_chunk), which is the
                # difference between finishing in ~1s and ~10min on 7B
                # configurations where ``N_chunk`` lands in the hundreds.
                #
                # Peak bound on (n_persist + n_buffer):
                #   int(alpha * (sum * S_chunk + F_bm)) <= capacity
                #   => sum <= floor((capacity/alpha - F_bm) / S_chunk)
                #
                # CAVEAT: this uses the legacy 1xS_chunk per-chunk
                # multiplier. Under full FT the true model-state cost
                # per persistent chunk is up to ~8x S_chunk (full Adam
                # state, see ``model_state_present_bytes``), so this
                # bound is OPTIMISTIC — it lets through more sums than
                # are actually feasible. That is safe (never excludes
                # a feasible config); the inner loop's tight GPU gate
                # via ``model_state_present_bytes`` does the real
                # rejection. The looseness only costs a few extra
                # inner iterations per (n_swap, n_ckpt, n_offload).
                #
                # CAVEAT: this bound uses the uncapped ``F_bm`` raw-peak
                # decomposition. The inner loop later applies
                # ``hot_iter_peak_cap`` which can LOWER ``raw_peak`` when
                # the per-block trace shows the F_bm op-walk overestimates
                # the true hot-iter peak. When the cap fires
                # (``raw_peak > hot_cap``), ``predicted_peak`` collapses to
                # ``alpha * hot_cap`` — independent of (n_persist+n_buffer).
                # If ``alpha * hot_cap <= capacity_bytes``, EVERY config
                # with sum > max_sum (which the F_bm bound would prune)
                # actually clears the GPU gate via the cap. Compute the cap
                # once per (n_swap, n_ckpt) pair — it depends only on
                # ``trace``, ``block_map``, and ``cfg.n_checkpoint``/
                # ``cfg.n_swap`` (see ``cost/memory.py::hot_iter_peak_cap``;
                # n_persist/n_buffer are not read) — and widen ``max_sum``
                # to the natural ``N_chunk`` ceiling when the cap rescues
                # the whole sum-axis. Probe cfg uses n_persist=n_buffer=0
                # because those fields are unused by ``hot_iter_peak_cap``.
                _cap_probe_cfg = CostConfig(
                    n_persist=0,
                    n_buffer=0,
                    n_swap=n_swap,
                    n_checkpoint=n_ckpt,
                    n_offload=n_offload,
                )
                _hot_cap = hot_iter_peak_cap(
                    trace, block_map, _cap_probe_cfg, layout=layout
                )
                _cap_dominates = (
                    _hot_cap is not None and int(alpha * _hot_cap) <= capacity_bytes
                )

                # F_bm depends on ``n_persist`` via the OFFLOAD-bump term:
                # ``_block_map_peak_contribution`` charges ``S_chunk`` per
                # OFFLOAD block at that block's last forward op, but the
                # runtime ``ChunkManager.gather`` short-circuits for
                # persistent chunks (``chunk/manager.py::gather`` "Persistent
                # chunks: no-op — they were never offloaded"). When an
                # OFFLOAD block's chunks are all in the persistent set the
                # backward-window chunk-buffer materialization does not
                # happen, so the bump must be suppressed. When ``n_offload``
                # is 0 the contribution is ``n_persist``-invariant and we
                # hoist a single computation outside the inner loop;
                # otherwise the inner loop recomputes per-``n_persist``.
                f_bm_invariant: int | None
                if n_offload == 0:
                    f_bm_invariant = _block_map_peak_contribution(
                        block_map,
                        trace,
                        layout,
                        forward_ops_by_block=forward_ops_by_block,
                        tree_index_map=tree_index_map,
                    )
                else:
                    f_bm_invariant = None

                for n_persist in range(0, bounds.N_chunk + 1):
                    # Recompute ``f_bm`` per ``n_persist`` when OFFLOAD
                    # blocks exist — the OFFLOAD bump drops out for blocks
                    # whose chunks are all persistent (see
                    # ``_block_map_peak_contribution`` docstring). When
                    # ``n_offload == 0`` the value is invariant and the
                    # hoisted ``f_bm_invariant`` is reused.
                    if f_bm_invariant is not None:
                        f_bm = f_bm_invariant
                    else:
                        f_bm = _block_map_peak_contribution(
                            block_map,
                            trace,
                            layout,
                            forward_ops_by_block=forward_ops_by_block,
                            tree_index_map=tree_index_map,
                            n_persist=n_persist,
                        )
                    if _cap_dominates:
                        max_sum = bounds.N_chunk
                    elif alpha > 0 and s_chunk > 0:
                        max_sum = int((capacity_bytes / alpha - f_bm) / s_chunk)
                    else:
                        max_sum = bounds.N_chunk
                    max_sum = max(0, min(max_sum, bounds.N_chunk))

                    # Max feasible n_buffer at this n_persist (partition + capacity).
                    max_buffer = min(bounds.N_chunk - n_persist, max_sum - n_persist)
                    if max_buffer < 0:
                        # n_persist alone exceeds the capacity budget at
                        # this ``f_bm``. With OFFLOAD active, future
                        # ``n_persist`` values may have a SMALLER ``f_bm``
                        # (more OFFLOAD blocks become fully persistent →
                        # fewer bumps survive), so the budget can re-open;
                        # use ``continue`` instead of ``break`` to keep
                        # scanning. With no OFFLOAD blocks the budget is
                        # monotone in n_persist and we can break.
                        if f_bm_invariant is not None:
                            break
                        continue

                    # Scheduler needs enough buffers to hold (current block's
                    # non-persistent chunks) union (next block's non-persistent
                    # chunks) simultaneously — that's how the lookahead
                    # prefetch in runtime/scheduler.py::pre_block_forward
                    # works. Skip n_persist values that can't support that
                    # minimum within the capacity budget.
                    min_buffer = min_n_buffer_for(layout, n_persist)
                    if min_buffer > max_buffer:
                        continue
                    if not block_map_runtime_admissible(layout, block_map, n_persist):
                        continue

                    # Optimum n_buffer is the max feasible: cached chunks
                    # skip re-gather in backward, and estimate_runtime is
                    # monotone non-increasing in n_buffer through the
                    # ``min(n_buffer, n_nonpersist)`` cache-hit term. We also
                    # evaluate n_buffer = min_buffer as the tie-break
                    # boundary so the picked config doesn't over-commit
                    # buffer capacity when the runtime is flat.
                    #
                    # When the CPU-RAM gate is active, the 2-point shortcut
                    # is unsound: ``max_buffer`` may fail the host-side
                    # ``estimate_cpu_footprint`` check (more buffered chunks
                    # = more pinned CPU staging) while an intermediate
                    # ``n_buffer`` is feasible AND faster than ``min_buffer``.
                    # Iterate the full feasible range in that case so we
                    # don't spuriously raise "no config fits" or pick a
                    # slower ``min_buffer`` config. Capacity bounds are
                    # unchanged — we still scan within ``[min_buffer,
                    # max_buffer]`` so the GPU gate stays enforced.
                    if cpu_capacity_bytes is None:
                        # Ordered tuple (min first) so tie-breaks prefer the
                        # smaller buffer — matches the searcher's
                        # strict ``<`` replacement rule below where the first
                        # candidate iterated wins on equal predicted cost.
                        n_buffer_candidates: Iterable[int] = (min_buffer, max_buffer)
                    else:
                        n_buffer_candidates = range(min_buffer, max_buffer + 1)
                    for n_buffer in n_buffer_candidates:
                        n_total += 1
                        cfg = CostConfig(
                            n_persist=n_persist,
                            n_buffer=n_buffer,
                            n_swap=n_swap,
                            n_checkpoint=n_ckpt,
                            n_offload=n_offload,
                        )
                        # Model-state residency must use the same
                        # persistent_factor / buffer_factor derivation
                        # as ``cost/memory.py::estimate_peak`` — the
                        # naive ``(n_persist + n_buffer) * S_chunk``
                        # form under-counts full Adam state on
                        # persistent chunks (8x S_chunk under fp16+Adam,
                        # not 1x), so the searcher's pruning would
                        # let through configs that ``estimate_peak``
                        # then rejects on the final validation pass.
                        # Single-source via ``model_state_present_bytes``
                        # so the two sites cannot drift again
                        # (regression follow-up to commit d908bf28).
                        model_state_present = model_state_present_bytes(
                            cfg, layout, trace
                        )
                        raw_peak = model_state_present + f_bm
                        # Apply the hot-iter ground-truth cap (v6+ traces with
                        # per-block peaks). Mirrors the cap in
                        # ``cost/memory.py::estimate_peak`` so the searcher
                        # picks the same config ``estimate_peak`` would
                        # validate, closing the F_bm-vs-estimate_peak gap.
                        _cap = hot_iter_peak_cap(
                            trace, block_map, cfg, layout=layout
                        )
                        if _cap is not None and raw_peak > _cap:
                            raw_peak = _cap
                        predicted_peak = int(alpha * raw_peak) if raw_peak > 0 else 0
                        if predicted_peak > capacity_bytes:
                            continue
                        n_gpu_feasible += 1
                        # Hard CPU-RAM feasibility gate. Skipped when
                        # ``cpu_capacity_bytes`` is None (caller opted out
                        # of host-side filtering — backward-compatible
                        # default). Estimated bytes are per-rank pinned
                        # CPU; sharding is reflected via hw.zero3_shard
                        # inside ``estimate_cpu_footprint``.
                        if cpu_capacity_bytes is not None:
                            cpu_footprint = estimate_cpu_footprint(
                                cfg, layout, hw, trace=trace
                            )
                            if cpu_footprint > cpu_capacity_bytes:
                                n_cpu_rejected += 1
                                continue
                        n_feasible += 1
                        predicted_iter_s = estimate_runtime(
                            cfg, trace, layout, block_map, hw
                        )
                        # Non-finite runtime (e.g. inf when CPU-Adam is
                        # unavailable for non-persistent chunks, or NaN from
                        # an underlying numerical failure) means this config
                        # cleared every capacity gate but cannot be costed.
                        # Track separately so the failure-mode disambiguator
                        # below doesn't blame GPU/CPU capacity when the real
                        # binding constraint is a runtime/dependency gap.
                        if not math.isfinite(predicted_iter_s):
                            n_runtime_rejected += 1
                            continue
                        if predicted_iter_s < best_iter_s:
                            best_iter_s = predicted_iter_s
                            best_cfg = cfg
                            best_block_map = block_map
                            best_peak = predicted_peak

    if best_cfg is None or best_block_map is None:
        # Disambiguate the failure mode for the caller. If every fully
        # capacity-feasible config produced a non-finite runtime
        # estimate, the binding constraint is a runtime/dependency gap
        # (e.g. CPU-Adam unavailable for non-persistent chunks), not
        # capacity — surface that explicitly so the user doesn't waste
        # time chasing memory budgets.
        if n_feasible > 0 and n_runtime_rejected == n_feasible:
            raise RuntimeError(
                "no ProTrain config has a finite runtime estimate; every "
                f"capacity-feasible config (out of {n_feasible}) was "
                "rejected by estimate_runtime (likely CPU-Adam unavailable "
                "for non-persistent chunks on this setup). Evaluated "
                f"{n_total} configs total."
            )
        # If at least one candidate cleared the GPU gate but every such
        # candidate exceeded the CPU envelope, the binding constraint is
        # host RAM, not GPU memory — surface that explicitly so the user
        # knows to add nodes / system RAM rather than larger cards.
        if (
            cpu_capacity_bytes is not None
            and n_gpu_feasible > 0
            and n_cpu_rejected == n_gpu_feasible
        ):
            raise RuntimeError(
                f"no ProTrain config fits in {cpu_capacity_bytes / 1e9:.1f} GB "
                f"host RAM (per-rank CPU budget); {n_gpu_feasible} configs "
                f"cleared the GPU capacity gate but all exceeded the CPU "
                f"footprint limit. Evaluated {n_total} configs total. "
                "Scale up: more nodes, more system RAM, or a smaller model."
            )
        raise RuntimeError(
            "no feasible ProTrain config under capacity_bytes="
            f"{capacity_bytes} (evaluated {n_total} configs)"
        )

    if cpu_capacity_bytes is not None:
        LOG.info(
            "ProTrain search: evaluated %d configs, %d cleared GPU gate, "
            "%d rejected by CPU gate, %d feasible, picked %s "
            "predicted=%dMB %.3fs (cpu_budget=%.1f GB)",
            n_total,
            n_gpu_feasible,
            n_cpu_rejected,
            n_feasible,
            best_cfg,
            best_peak // (1 << 20),
            best_iter_s,
            cpu_capacity_bytes / 1e9,
        )
    else:
        LOG.info(
            "ProTrain search: evaluated %d configs, %d feasible, picked %s "
            "predicted=%dMB %.3fs",
            n_total,
            n_feasible,
            best_cfg,
            best_peak // (1 << 20),
            best_iter_s,
        )
    return SearchResult(
        cfg=best_cfg,
        block_map=best_block_map,
        predicted_peak_bytes=best_peak,
        predicted_iter_s=best_iter_s,
    )


__all__ = [
    "block_map_runtime_admissible",
    "min_n_buffer_for",
    "search",
]
