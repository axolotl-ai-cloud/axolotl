"""Block-granularity runtime scheduler: prefetch/release/reduce-offload at block boundaries."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Iterable

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

    from axolotl.integrations.protrain.chunk import ChunkManager

LOG = get_logger(__name__)


def _step_timing_enabled() -> bool:
    """PROTRAIN_DEBUG_STEP_TIMING={1,true,yes,on} enables per-block-method timing aggregation."""
    raw = os.environ.get("PROTRAIN_DEBUG_STEP_TIMING", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _step_timing_emit_every() -> int:
    """PROTRAIN_DEBUG_STEP_TIMING_EVERY (default 50) controls aggregate emit cadence."""
    raw = os.environ.get("PROTRAIN_DEBUG_STEP_TIMING_EVERY", "50")
    try:
        n = int(raw)
    except ValueError:
        return 50
    return max(1, n)


def _first_iter_trace_disabled() -> bool:
    """PROTRAIN_DEBUG_FIRST_ITER_TRACE=0 disables the auto first-iter trace (default: enabled)."""
    raw = os.environ.get("PROTRAIN_DEBUG_FIRST_ITER_TRACE", "").strip().lower()
    return raw in ("0", "false", "no", "off")


class Scheduler:
    """Drives prefetch / release / reduce-offload at block granularity."""

    def __init__(
        self,
        chunk_manager: "ChunkManager",
        block_map: BlockStrategyMap,
        layout: ChunkLayout,
        effective_h2d_bps: float,
        effective_d2h_bps: float,
    ) -> None:
        self.chunk_manager = chunk_manager
        self.block_map = block_map
        self.layout = layout
        self.effective_h2d_bps = float(effective_h2d_bps)
        self.effective_d2h_bps = float(effective_d2h_bps)

        # Forward-order block list; encoder ids before decoder ids by construction.
        self._block_order: list[BlockId] = sorted(block_map.keys())
        # O(1) reverse lookup for next/prev_block_of on deep models.
        self._block_index_map: dict[BlockId, int] = {
            block_id: idx for idx, block_id in enumerate(self._block_order)
        }

        # Earliest-forward owner = last in backward; defer reduce until then for shared-chunk blocks.
        chunk_last_bwd_owner: dict[ChunkId, BlockId] = {}
        for bid, cids in self.layout.block_to_chunks.items():
            for cid in cids:
                prev = chunk_last_bwd_owner.get(cid)
                if prev is None or bid < prev:
                    chunk_last_bwd_owner[cid] = bid
        self._chunk_last_bwd_owner: dict[ChunkId, BlockId] = chunk_last_bwd_owner

        # bs=1 hot-path: precompute per-block walk metadata so per-step hooks
        # only do dict lookups, not list comprehensions / set constructions.
        # Indexed by BlockId in self._block_order.
        # next_chunks_set[bid] = frozenset of chunks owned by the *next* block
        # (used by post_block_forward to skip releasing chunks the next block
        # still needs). next_block_of_cached[bid] = next block id or None.
        # prev_block_of_cached[bid] = previous block id or None (for backward).
        self._next_block_of_cached: dict[BlockId, "BlockId | None"] = {}
        self._prev_block_of_cached: dict[BlockId, "BlockId | None"] = {}
        self._next_chunks_set_cached: dict[BlockId, frozenset[ChunkId]] = {}
        # Pre-filtered tuple of chunks this block finalizes in post_block_backward;
        # avoids the per-step _chunk_last_bwd_owner dict lookup + identity compare.
        self._owned_chunks_for_finalize_cached: dict[BlockId, tuple[ChunkId, ...]] = {}
        for idx, bid in enumerate(self._block_order):
            nxt = (
                self._block_order[idx + 1] if idx + 1 < len(self._block_order) else None
            )
            prv = self._block_order[idx - 1] if idx >= 1 else None
            self._next_block_of_cached[bid] = nxt
            self._prev_block_of_cached[bid] = prv
            nxt_chunks = (
                self.layout.block_to_chunks.get(nxt, ()) if nxt is not None else ()
            )
            self._next_chunks_set_cached[bid] = frozenset(nxt_chunks)
            owned = tuple(
                cid
                for cid in self.layout.block_to_chunks.get(bid, ())
                if self._chunk_last_bwd_owner.get(cid, bid) == bid
            )
            self._owned_chunks_for_finalize_cached[bid] = owned

        self._prefetch_stream: "torch.cuda.Stream | None" = None
        self._swap_stream: "torch.cuda.Stream | None" = None
        # Dedicated stream for backward OFFLOAD re-gather (H2D + NCCL all_gather).
        # Mirror of _swap_stream; lets re-gather of block N-1 overlap backward
        # compute of block N on non-NVLink topology where the compute stream
        # would otherwise serialize with the PCIe copy + collective.
        # Only created when n_offload > 0; None otherwise (no overhead on inert
        # / no-offload configs).
        self._offload_stream: "torch.cuda.Stream | None" = None
        # Lazy ActivationSwapPool attach by wrapper when n_swap > 0.
        self.swap_pool: object | None = None
        self._closed: bool = False
        # bs=1 hot-path cache: ``cuda.is_available()`` costs ~2.7us per call
        # and is invariant for the scheduler's lifetime. Caching as a bool
        # lets per-step LoRA-container hooks dispatch on an attribute lookup
        # instead of a CUDA syscall (§16 PR #4).
        self._has_cuda: bool = False
        # Set by install_hooks when the runtime is provably inert (all chunks
        # persistent, no OFFLOAD, modes NONE/CKPT). Per-step methods short-
        # circuit so stragglers (e.g. recompute callbacks) skip stream-wait
        # syscalls that would no-op anyway.
        self._is_inert: bool = False

        # Per-step timing diagnostics. Opt-in via PROTRAIN_DEBUG_STEP_TIMING=1
        # to avoid the perf_counter overhead on hot configs.
        self._step_timing_enabled: bool = _step_timing_enabled()
        self._step_timing_emit_every: int = _step_timing_emit_every()
        # Sum and call-count per method name across pre_block_forward,
        # post_block_forward, pre_block_backward, post_block_backward,
        # ensure_block_resident, ensure_chunks_resident. Reset on each emit.
        self._step_timing_sum_ns: dict[str, int] = {}
        self._step_timing_calls: dict[str, int] = {}
        # Step boundaries are inferred from drain() calls; each drain emits
        # the running aggregate if the cadence threshold is met.
        self._step_timing_step_idx: int = 0
        if self._step_timing_enabled:
            LOG.info(
                "ProTrain Scheduler: PROTRAIN_DEBUG_STEP_TIMING enabled "
                "(emit every %d step(s)). Per-method aggregate wall-time "
                "will be logged at INFO so it survives DEBUG suppression.",
                self._step_timing_emit_every,
            )

        # First-iter trace: logs per-block fwd/bwd entry+exit with wall-clock
        # elapsed since iter start. Auto-disables after drain() fires once.
        # PR #15 (n_offload>0 first-iter hang diagnostic): bs=2 Mode B with
        # n_offload=32 hit a >254s hang on the first iteration during v71
        # hardware verification, but GPU was 100% utilized so the loss was
        # somewhere between block hooks, not in CPU code. This trace logs at
        # INFO so the gap between two adjacent log lines pinpoints which
        # block / which hook held the hang.
        self._first_iter_trace_enabled: bool = not _first_iter_trace_disabled()
        self._first_iter_t0_ns: int = 0
        if self._first_iter_trace_enabled:
            LOG.info(
                "ProTrain Scheduler: first-iter trace enabled (set "
                "PROTRAIN_DEBUG_FIRST_ITER_TRACE=0 to disable). Per-block "
                "fwd/bwd entry timestamps will be logged at INFO during the "
                "first training iteration; the trace self-disables after the "
                "first drain() so hot-path overhead is bounded to iter 1."
            )
        self._init_streams()

    @property
    def swap_stream(self) -> "torch.cuda.Stream | None":
        """Public accessor for the dedicated activation-swap stream (None on CPU)."""
        return self._swap_stream

    @property
    def offload_stream(self) -> "torch.cuda.Stream | None":
        """Public accessor for the dedicated OFFLOAD backward-regather stream (None when n_offload == 0 or CPU)."""
        return self._offload_stream

    def _has_offloaded_blocks(self) -> bool:
        """True iff any block in the strategy map is in OFFLOAD mode (needs backward re-gather)."""
        for mode in self.block_map.values():
            if mode is BlockMode.OFFLOAD:
                return True
        return False

    def _init_streams(self) -> None:
        """Create dedicated CUDA streams for prefetch + activation swap + OFFLOAD re-gather."""
        try:
            import torch
        except ImportError:  # pragma: no cover — torch is required at runtime
            return

        self._has_cuda = bool(torch.cuda.is_available())
        if not self._has_cuda:
            LOG.debug(
                "Scheduler: CUDA unavailable; prefetch/swap/offload streams are None "
                "(scheduler degrades to synchronous transfers)."
            )
            self._prefetch_stream = None
            self._swap_stream = None
            self._offload_stream = None
            return

        # Non-default stream lets compute overlap PCIe copies.
        self._prefetch_stream = torch.cuda.Stream()
        # Separate swap stream avoids contention with chunk prefetch.
        self._swap_stream = torch.cuda.Stream()
        # Dedicated OFFLOAD re-gather stream — only when at least one block is OFFLOAD.
        # Without this, per-chunk H2D + NCCL all_gather during backward run on the
        # compute stream and serialize with backward compute (bs=2 hang on
        # non-NVLink topology). Skipping creation for n_offload == 0 keeps inert
        # configs free of any per-step wait_stream overhead.
        if self._has_offloaded_blocks():
            self._offload_stream = torch.cuda.Stream()
        else:
            self._offload_stream = None

    # ---- helpers -------------------------------------------------------

    def _record_step_timing(self, name: str, elapsed_ns: int) -> None:
        """Aggregate one wall-time sample for ``name`` into the running totals."""
        self._step_timing_sum_ns[name] = (
            self._step_timing_sum_ns.get(name, 0) + elapsed_ns
        )
        self._step_timing_calls[name] = self._step_timing_calls.get(name, 0) + 1

    def _emit_step_timing(self) -> None:
        """Log + reset the aggregate, called from drain() once per step boundary."""
        if not self._step_timing_sum_ns:
            return
        total_ns = sum(self._step_timing_sum_ns.values())
        # INFO so the diagnostic survives default log levels; cost gated by env var.
        LOG.info(
            "ProTrain Scheduler step-timing aggregate (steps_seen=%d, "
            "emit_every=%d, total_ms=%.3f): %s",
            self._step_timing_step_idx,
            self._step_timing_emit_every,
            total_ns / 1e6,
            {
                name: {
                    "calls": self._step_timing_calls.get(name, 0),
                    "sum_ms": self._step_timing_sum_ns.get(name, 0) / 1e6,
                    "avg_us": (
                        (self._step_timing_sum_ns.get(name, 0) / 1e3)
                        / max(1, self._step_timing_calls.get(name, 0))
                    ),
                }
                for name in sorted(self._step_timing_sum_ns)
            },
        )
        self._step_timing_sum_ns.clear()
        self._step_timing_calls.clear()

    def _first_iter_log(self, event: str, block_id: "BlockId | None" = None) -> None:
        """Log ``event`` at INFO with wall-clock elapsed since iter start. Hot-path-safe (one branch + format)."""
        if not self._first_iter_trace_enabled:
            return
        now = time.perf_counter_ns()
        if self._first_iter_t0_ns == 0:
            self._first_iter_t0_ns = now
        elapsed_ms = (now - self._first_iter_t0_ns) / 1e6
        if block_id is None:
            LOG.info(
                "ProTrain first-iter trace: t+%.1fms %s",
                elapsed_ms,
                event,
            )
        else:
            LOG.info(
                "ProTrain first-iter trace: t+%.1fms block=%d %s",
                elapsed_ms,
                int(block_id),
                event,
            )

    def _chunks_for(self, block_id: BlockId) -> tuple[ChunkId, ...]:
        """Return the chunks owned by ``block_id`` under the current layout."""
        return self.layout.block_to_chunks.get(block_id, ())

    def _next_block_of(self, block_id: BlockId) -> BlockId | None:
        """Return the block id scheduled *after* ``block_id`` in forward order."""
        return self._next_block_of_cached.get(block_id)

    def _prev_block_of(self, block_id: BlockId) -> BlockId | None:
        """Return the next-in-backward block (idx-1 in forward order)."""
        return self._prev_block_of_cached.get(block_id)

    def _gather_on_prefetch_stream(self, chunk_ids: Iterable[ChunkId]) -> None:
        """Async-gather chunk_ids on the prefetch stream; waits on swap_stream to sequence SWAP D2H."""
        if self._prefetch_stream is None or not self._has_cuda:
            # Synchronous fallback.
            for cid in chunk_ids:
                self.chunk_manager.gather(cid)
            return

        import torch

        # Order H2D writes after in-flight SWAP D2H reads (correctness for SWAP × non-persistent).
        if self._swap_stream is not None:
            self._prefetch_stream.wait_stream(self._swap_stream)

        with torch.cuda.stream(self._prefetch_stream):
            for cid in chunk_ids:
                self.chunk_manager.gather(cid)

    def _sync_prefetch_with_compute(self) -> None:
        """Make the default compute stream wait on the prefetch stream."""
        if self._prefetch_stream is None or not self._has_cuda:
            return
        import torch

        compute = torch.cuda.current_stream()
        compute.wait_stream(self._prefetch_stream)

    def _gather_on_offload_stream(self, chunk_ids: Iterable[ChunkId]) -> None:
        """Async-regather offloaded chunk_ids on the OFFLOAD stream during backward.

        Mirrors :meth:`_gather_on_prefetch_stream` but targets ``_offload_stream``
        so per-chunk H2D copy + NCCL all_gather overlap backward compute on the
        default compute stream. Sequences after prefetch (any in-flight forward
        gathers) and after swap (D2H reads) for correctness.
        """
        if self._offload_stream is None or not self._has_cuda:
            for cid in chunk_ids:
                self.chunk_manager.gather(cid, phase="backward_regather")
            return

        import torch

        # Sequence after any in-flight forward gathers + activation swap D2H.
        if self._prefetch_stream is not None:
            self._offload_stream.wait_stream(self._prefetch_stream)
        if self._swap_stream is not None:
            self._offload_stream.wait_stream(self._swap_stream)

        with torch.cuda.stream(self._offload_stream):
            for cid in chunk_ids:
                self.chunk_manager.gather(cid, phase="backward_regather")

    def _sync_offload_with_compute(self) -> None:
        """Make the default compute stream wait on the OFFLOAD re-gather stream.

        Backward compute on block N must not start reading the chunk's data
        before the H2D + NCCL all_gather scheduled on ``_offload_stream`` has
        completed.
        """
        if self._offload_stream is None or not self._has_cuda:
            return
        import torch

        compute = torch.cuda.current_stream()
        compute.wait_stream(self._offload_stream)

    def ensure_block_resident(self, block_id: BlockId) -> None:
        """Sync gather block's chunks; used by checkpoint recompute path that bypasses pre-hooks."""
        _t0 = time.perf_counter_ns() if self._step_timing_enabled else 0
        try:
            if self._is_inert:
                return
            chunk_ids = self._chunks_for(block_id)
            if not chunk_ids:
                return
            # OFFLOAD blocks route re-gather via _offload_stream so the per-chunk
            # H2D + NCCL all_gather overlaps backward compute instead of
            # serializing with it on the compute stream.
            mode = self.block_map.get(block_id, BlockMode.NONE)
            if mode is BlockMode.OFFLOAD and self._offload_stream is not None:
                self._gather_on_offload_stream(chunk_ids)
                self._sync_offload_with_compute()
            else:
                self._gather_on_prefetch_stream(chunk_ids)
                self._sync_prefetch_with_compute()
        finally:
            if self._step_timing_enabled:
                self._record_step_timing(
                    "ensure_block_resident", time.perf_counter_ns() - _t0
                )

    def ensure_chunks_resident(self, chunk_ids: Iterable[ChunkId]) -> None:
        """Gather chunks for LoRA-container hooks; routes sharded all_gather on prefetch stream so it overlaps compute.

        ``param.data`` rebinding inside :meth:`ChunkManager.gather` is a
        CPU-side tensor metadata swap that completes synchronously, so
        autograd sees real ``param.size()`` immediately. The H2D copy +
        NCCL ``all_gather_into_tensor`` are stream-dependent; routing them
        on ``_prefetch_stream`` (instead of the compute stream) avoids the
        Mode C ``zero3_shard`` hang where block N's compute would otherwise
        serialize with block N-1's per-container sharded gather.
        """
        # Per-step LoRA-container fan-out (~28 containers × 4 hooks at Llama
        # class) drives this method hot — keep the CPU lane an attribute-
        # lookup-only no-op past the empty-tuple guard. The cuda.is_available
        # check is read off the cached scheduler flag, not the syscall.
        _t0 = time.perf_counter_ns() if self._step_timing_enabled else 0
        try:
            if self._is_inert:
                return
            if isinstance(chunk_ids, tuple):
                cids = chunk_ids
            else:
                cids = tuple(chunk_ids)
            if not cids:
                return
            if self._has_cuda and self._prefetch_stream is not None:
                import torch as _torch

                # Order prefetch after in-flight swap/offload so pool buffers
                # and any concurrent re-gather complete before we issue ours.
                if self._swap_stream is not None:
                    self._prefetch_stream.wait_stream(self._swap_stream)
                if self._offload_stream is not None:
                    self._prefetch_stream.wait_stream(self._offload_stream)
                # Issue H2D + NCCL all_gather on the prefetch stream so the
                # sharded reconstruction overlaps with current compute on the
                # default stream.
                with _torch.cuda.stream(self._prefetch_stream):
                    for cid in cids:
                        self.chunk_manager.gather(cid)
                # Compute must observe the gather's writes before reading the
                # chunk; mirror _sync_prefetch_with_compute inline so the
                # caller's compute stream blocks on prefetch.
                compute = _torch.cuda.current_stream()
                compute.wait_stream(self._prefetch_stream)
                return
            # CPU / no-prefetch-stream fallback: synchronous gather (correctness).
            for cid in cids:
                self.chunk_manager.gather(cid)
        finally:
            if self._step_timing_enabled:
                self._record_step_timing(
                    "ensure_chunks_resident", time.perf_counter_ns() - _t0
                )

    # ---- forward -------------------------------------------------------

    def pre_block_forward(self, block_id: BlockId) -> None:
        """Prefetch the next block's chunks; ensure current block's are resident."""
        _t0 = time.perf_counter_ns() if self._step_timing_enabled else 0
        if self._first_iter_trace_enabled:
            self._first_iter_log("pre_block_forward enter", block_id)
        try:
            if self._is_inert:
                return
            # gather() is idempotent on persistent / already-resident chunks.
            self.ensure_block_resident(block_id)

            # Async prefetch next block; DO NOT sync — copy overlaps current compute.
            nxt = self._next_block_of(block_id)
            if nxt is None:
                return
            next_chunks = self._chunks_for(nxt)
            if not next_chunks:
                return
            self._gather_on_prefetch_stream(next_chunks)
            LOG.debug(
                "Scheduler.pre_block_forward: block=%d prefetched %d chunks for next block %d",
                block_id,
                len(next_chunks),
                nxt,
            )
        finally:
            if self._first_iter_trace_enabled:
                self._first_iter_log("pre_block_forward exit", block_id)
            if self._step_timing_enabled:
                self._record_step_timing(
                    "pre_block_forward", time.perf_counter_ns() - _t0
                )

    def post_block_forward(self, block_id: BlockId) -> None:
        """Release this block's non-persistent chunks except those used by the next block."""
        _t0 = time.perf_counter_ns() if self._step_timing_enabled else 0
        if self._first_iter_trace_enabled:
            self._first_iter_log("post_block_forward enter", block_id)
        try:
            if self._is_inert:
                return
            # frozenset from the init-time cache; avoids the per-step set() build.
            next_chunks = self._next_chunks_set_cached.get(block_id, frozenset())
            for cid in self._chunks_for(block_id):
                if cid in next_chunks:
                    continue
                # offload() short-circuits for persistent chunks.
                self.chunk_manager.offload(cid)
        finally:
            if self._first_iter_trace_enabled:
                self._first_iter_log("post_block_forward exit", block_id)
            if self._step_timing_enabled:
                self._record_step_timing(
                    "post_block_forward", time.perf_counter_ns() - _t0
                )

    # ---- backward ------------------------------------------------------

    def pre_block_backward(self, block_id: BlockId) -> None:
        """Ensure block's chunks are resident before backward; cover chunk-state path only (SWAP handles activations)."""
        _t0 = time.perf_counter_ns() if self._step_timing_enabled else 0
        if self._first_iter_trace_enabled:
            self._first_iter_log("pre_block_backward enter", block_id)
        try:
            self._pre_block_backward_impl(block_id)
        finally:
            if self._first_iter_trace_enabled:
                self._first_iter_log("pre_block_backward exit", block_id)
            if self._step_timing_enabled:
                self._record_step_timing(
                    "pre_block_backward", time.perf_counter_ns() - _t0
                )

    def _pre_block_backward_impl(self, block_id: BlockId) -> None:
        """Untimed pre_block_backward body; split out to keep the timed wrapper readable."""
        if self._is_inert:
            return
        mode = self.block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.SWAP:
            LOG.debug(
                "Scheduler.pre_block_backward: block=%d is SWAP; "
                "activation H2D scheduled by SwappedBlock on swap_stream",
                block_id,
            )
        elif mode is BlockMode.OFFLOAD:
            # OFFLOAD: pre-warm chunk so unpack hook hits resident fast-path.
            LOG.debug(
                "Scheduler.pre_block_backward: block=%d is OFFLOAD; "
                "pre-warming chunk for saved-tensor unpack hook",
                block_id,
            )

        chunk_ids = self._chunks_for(block_id)
        if not chunk_ids:
            return

        # All-persistent layouts skip pool construction; bail before pool-cache NPE.
        if self.chunk_manager.buffer_pool is None:
            return

        # Resident tag means slot is assigned but H2D may still be in flight; sync prefetch first.
        self._sync_prefetch_with_compute()
        # OFFLOAD re-gather from a prior block must complete before backward
        # reads chunk data; mirrors _sync_prefetch_with_compute for the
        # dedicated offload stream.
        self._sync_offload_with_compute()

        is_offload = mode is BlockMode.OFFLOAD and self._offload_stream is not None

        # Consult pool: hits are free, misses trigger fresh H2D on prefetch (or offload) stream.
        misses: list[ChunkId] = []
        for cid in chunk_ids:
            if self.chunk_manager.buffer_pool.lookup_resident(cid) is None:
                misses.append(cid)
            else:
                # Re-claim the slot (removes from free list if present).
                self.chunk_manager.gather(cid, phase="backward_regather")
        if misses:
            if is_offload:
                self._gather_on_offload_stream(misses)
                self._sync_offload_with_compute()
            else:
                self._gather_on_prefetch_stream(misses)
                self._sync_prefetch_with_compute()

        # Mirror forward look-ahead: async prefetch next-in-backward block's chunks.
        nxt_bwd = self._prev_block_of(block_id)
        if nxt_bwd is None:
            return
        nxt_chunks = self._chunks_for(nxt_bwd)
        if not nxt_chunks:
            return
        need = [
            cid
            for cid in nxt_chunks
            if self.chunk_manager.buffer_pool.lookup_resident(cid) is None
        ]
        if need:
            # Route next-in-backward look-ahead via the offload stream when the
            # upcoming block is OFFLOAD; otherwise stay on the prefetch stream
            # (SWAP / NONE blocks).
            nxt_mode = self.block_map.get(nxt_bwd, BlockMode.NONE)
            if nxt_mode is BlockMode.OFFLOAD and self._offload_stream is not None:
                self._gather_on_offload_stream(need)
            else:
                self._gather_on_prefetch_stream(need)

    def post_block_backward(self, block_id: BlockId) -> None:
        """Finalize block's backward: release buffers + maybe kick CPU Adam."""
        _t0 = time.perf_counter_ns() if self._step_timing_enabled else 0
        if self._first_iter_trace_enabled:
            self._first_iter_log("post_block_backward enter", block_id)
        try:
            if self._is_inert:
                return
            # Pre-filtered at __init__ to skip the per-chunk owner lookup; the
            # shared-chunk filter (only earliest-forward owner finalizes) is
            # baked into the cached tuple.
            for cid in self._owned_chunks_for_finalize_cached.get(block_id, ()):
                self.chunk_manager.reduce_grads_and_offload(cid)
        finally:
            if self._first_iter_trace_enabled:
                self._first_iter_log("post_block_backward exit", block_id)
            if self._step_timing_enabled:
                self._record_step_timing(
                    "post_block_backward", time.perf_counter_ns() - _t0
                )

    # ---- end-of-iteration cleanup -------------------------------------

    def drain(self) -> None:
        """Block until every in-flight CPU Adam step has finished; flush deferred offloads."""
        if self._first_iter_trace_enabled:
            self._first_iter_log("drain enter (stream sync + cpu_optim drain)")

        # Drain in-flight prefetch/swap/offload traffic for stable peak-memory stats.
        if self._has_cuda:
            if self._prefetch_stream is not None:
                self._prefetch_stream.synchronize()
            if self._swap_stream is not None:
                self._swap_stream.synchronize()
            if self._offload_stream is not None:
                self._offload_stream.synchronize()

        if self._first_iter_trace_enabled:
            self._first_iter_log("drain post stream-sync; pre drain_deferred_offloads")

        # Defensive end-of-iter drain (only refcount==0 entries fire).
        self.chunk_manager.drain_deferred_offloads()

        if self._first_iter_trace_enabled:
            self._first_iter_log("drain post drain_deferred_offloads; pre wait_cpu_optim")

        self.chunk_manager.wait_cpu_optim()

        if self._first_iter_trace_enabled:
            self._first_iter_log("drain exit (iter 1 complete; trace auto-disabled)")
            # Self-disable so iter 2+ pays no first-iter-trace overhead.
            self._first_iter_trace_enabled = False

        # Step boundary; emit aggregate every Nth call when timing is enabled.
        if self._step_timing_enabled:
            self._step_timing_step_idx += 1
            if self._step_timing_step_idx % self._step_timing_emit_every == 0:
                self._emit_step_timing()

    # ---- teardown -----------------------------------------------------

    def close(self) -> None:
        """Synchronize + drop streams and swap pool. Idempotent. Does NOT close chunk manager."""
        if self._closed:
            return
        self._closed = True
        if self._has_cuda:
            for stream in (
                self._prefetch_stream,
                self._swap_stream,
                self._offload_stream,
            ):
                if stream is None:
                    continue
                try:
                    stream.synchronize()
                except Exception as exc:  # noqa: BLE001 — best-effort
                    LOG.debug("Scheduler.close: stream synchronize failed: %s", exc)
        self._prefetch_stream = None
        self._swap_stream = None
        self._offload_stream = None
        if self.swap_pool is not None:
            try:
                self.swap_pool.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("Scheduler.close: swap_pool.close failed: %s", exc)
            self.swap_pool = None


__all__ = ["Scheduler"]
