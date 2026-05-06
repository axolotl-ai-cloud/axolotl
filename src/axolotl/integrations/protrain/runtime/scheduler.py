"""Block-granularity runtime scheduler (§5, §6).

The :class:`Scheduler` sits between the transformer-block hooks (see
:mod:`axolotl.integrations.protrain.runtime.hooks`) and the chunk
manager. Its four entry points mirror the four lifecycle edges of a
transformer block:

* :meth:`pre_block_forward` — prefetch the **next** block's chunks so
  they are resident by the time compute reaches them.
* :meth:`post_block_forward` — release buffers whose last forward use
  was this block (keeping the next block's buffers resident for reuse).
* :meth:`pre_block_backward` — ensure this block's chunks are resident
  (re-gathering only if the forward-cached buffer was evicted).
* :meth:`post_block_backward` — reduce-offload this block's chunk
  gradients; this kicks off the CPU FusedAdam step asynchronously.

Stream policy
-------------
Prefetch and gather traffic runs on a dedicated *prefetch stream*
distinct from the default compute stream. Correctness is guaranteed at
block boundaries by synchronising the prefetch stream onto the current
(compute) stream before control returns to the caller — perfect overlap
is a pleasant side-effect when the kernels happen to run long enough,
but the scheduler never *relies* on it (the cost model did).

Activation swap is gated by the block wrapper (see
:class:`~axolotl.integrations.protrain.block.swap.SwappedBlock`); for
SWAP blocks the scheduler only has to keep the chunk-state path
consistent — the SWAP wrapper handles the activation copy itself.
"""

from __future__ import annotations

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


class Scheduler:
    """Drives prefetch / release / reduce-offload at block granularity.

    Parameters
    ----------
    chunk_manager:
        Runtime chunk driver; the scheduler never allocates buffers
        directly — it only calls ``gather`` / ``offload`` /
        ``reduce_grads_and_offload`` on the manager.
    block_map:
        Per-block activation mode (NONE / CKPT / SWAP) chosen by the
        searcher. Scheduler consults this to decide whether SWAP-specific
        prefetch paths need to be poked for backward.
    layout:
        The :class:`ChunkLayout` whose ``block_to_chunks`` dict tells
        the scheduler which chunks belong to which block.
    effective_h2d_bps / effective_d2h_bps:
        Post-contention effective bandwidths. Not consumed by M4b itself
        (the plan checks overlap at block boundaries, not per-transfer)
        but stored for the telemetry path in M5 and to surface the
        scheduler's current budget to callers.
    """

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

        # Ordered list of block ids — matches forward traversal order
        # by construction (``flatten_block_trees(discover_blocks(...))``
        # emits encoder ids before decoder ids; sorted(block_map.keys())
        # therefore reproduces the forward traversal order on both
        # single-tree and encoder-decoder models). Used to resolve
        # "next block" for the prefetch rule.
        self._block_order: list[BlockId] = sorted(block_map.keys())
        # O(1) reverse lookup of forward-order index for each block id;
        # avoids the O(n) ``list.index()`` scan in ``_next_block_of`` /
        # ``_prev_block_of`` on deep models (e.g., 96-layer).
        self._block_index_map: dict[BlockId, int] = {
            block_id: idx for idx, block_id in enumerate(self._block_order)
        }

        # Precompute the "last backward owner" for every chunk. ``build_layout``
        # packs the params of one transformer block into one chunk when they
        # fit, but a single chunk can hold params of MULTIPLE consecutive
        # blocks under the block-contiguity rule (§3.1.1). When that happens
        # ``layout.block_to_chunks[bid]`` for two adjacent blocks both contain
        # the shared chunk id. Backward iterates blocks in reverse-forward
        # order, so the LATER block in forward visits the chunk FIRST in
        # backward — if ``post_block_backward`` calls
        # ``reduce_grads_and_offload(cid)`` then, the chunk's grads are
        # finalized before the EARLIER block has produced its grads, which
        # at best wastes a regather/offload cycle and at worst
        # double-finalizes the chunk's reduce / CPU-optim state. Defer the
        # finalize until the EARLIEST forward-order block that owns each
        # chunk runs its post-backward — that block is the LAST visit in
        # backward. ``BlockId`` is a ``NewType("BlockId", int)`` and the
        # scheduler already assumes forward order is ascending integer order
        # (see ``_block_order = sorted(block_map.keys())`` above), so the
        # earliest-forward owner is simply ``min(owners)``.
        chunk_last_bwd_owner: dict[ChunkId, BlockId] = {}
        for bid, cids in self.layout.block_to_chunks.items():
            for cid in cids:
                prev = chunk_last_bwd_owner.get(cid)
                if prev is None or bid < prev:
                    chunk_last_bwd_owner[cid] = bid
        self._chunk_last_bwd_owner: dict[ChunkId, BlockId] = chunk_last_bwd_owner

        self._prefetch_stream: "torch.cuda.Stream | None" = None
        self._swap_stream: "torch.cuda.Stream | None" = None
        # ActivationSwapPool reference, attached lazily by the model
        # wrapper when ``n_swap > 0``. Type-erased to ``object`` here so
        # the scheduler module does not depend on ``block.swap_pool``.
        self.swap_pool: object | None = None
        self._init_streams()

    @property
    def swap_stream(self) -> "torch.cuda.Stream | None":
        """Public accessor for the dedicated activation-swap stream.

        Returned for the model wrapper to thread into each
        :class:`SwappedBlock` via :meth:`SwappedBlock.attach_runtime`.
        ``None`` on CPU-only paths.
        """
        return self._swap_stream

    def _init_streams(self) -> None:
        """Create dedicated CUDA streams for prefetch + activation swap.

        Two independent non-default streams: one for chunk prefetch
        (parameters), one for activation D2H/H2D under SWAP. Keeping
        them separate lets the chunk gather for block N+1 overlap with
        the activation H2D for block N during backward — the same
        single-block lookahead pattern the chunk prefetch already uses.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover — torch is required at runtime
            return

        if not torch.cuda.is_available():
            LOG.debug(
                "Scheduler: CUDA unavailable; prefetch/swap streams are None "
                "(scheduler degrades to synchronous transfers)."
            )
            self._prefetch_stream = None
            self._swap_stream = None
            return

        # A non-default stream lets the allocator / kernel launches on
        # the compute stream continue while PCIe copies are in flight.
        self._prefetch_stream = torch.cuda.Stream()
        # Activation SWAP runs on its own stream so D2H/H2D from the
        # block wrapper does not contend with chunk prefetch traffic.
        # Even on PCIe-bound 3090s where overlap with compute is
        # limited, isolating the streams keeps the cost model honest
        # (it already assumes the swap stream is independent).
        self._swap_stream = torch.cuda.Stream()

    # ---- helpers -------------------------------------------------------

    def _chunks_for(self, block_id: BlockId) -> tuple[ChunkId, ...]:
        """Return the chunks owned by ``block_id`` under the current layout."""
        return self.layout.block_to_chunks.get(block_id, ())

    def _next_block_of(self, block_id: BlockId) -> BlockId | None:
        """Return the block id scheduled *after* ``block_id`` in forward order."""
        idx = self._block_index_map.get(block_id)
        if idx is None:
            return None
        nxt = idx + 1
        if nxt >= len(self._block_order):
            return None
        return self._block_order[nxt]

    def _prev_block_of(self, block_id: BlockId) -> BlockId | None:
        """Return the block id scheduled *after* ``block_id`` in backward order.

        Backward walks the block list in reverse, so the "next" block in
        backward is the one with index ``idx - 1`` in forward order.
        """
        idx = self._block_index_map.get(block_id)
        if idx is None or idx <= 0:
            return None
        return self._block_order[idx - 1]

    def _gather_on_prefetch_stream(self, chunk_ids: Iterable[ChunkId]) -> None:
        """Async-gather ``chunk_ids`` on the prefetch stream.

        No-op if the prefetch stream is unavailable (CPU-only test
        lanes) — the chunk manager's synchronous ``gather`` is still
        correct; it is simply serialised against compute.

        App B.2 wire-up note: the ``with torch.cuda.stream(self._prefetch_stream)``
        block below sets the *kernel-launch* stream to the prefetch
        stream so that H2D copies overlap compute. Any *allocations*
        the chunk manager makes inside the gather call route through
        :class:`SingleStreamAllocator` (see ``chunk/manager.py::_gather_sharded``
        and the buffer-pool pre-allocation in
        ``chunk/buffer_pool.py::BufferPool.__init__``) so the bytes
        come from the default-stream heap regardless of which stream
        is current here. ``record_stream`` calls inside the chunk
        manager tie those buffers' lifetimes to this prefetch stream
        for correctness.

        SWAP-stream coordination (paper-fidelity §3.5+): when a SWAP
        block's pack hook has enqueued a D2H copy on ``self._swap_stream``
        reading FROM a chunk buffer slot, the next ``acquire`` on that
        slot may issue a fresh H2D writing INTO the same bytes via this
        method. Without an explicit cross-stream dependency the two
        transfers are unordered and the slot's pre-evict bytes (which
        SWAP's pack must capture intact for backward correctness) can
        be overwritten mid-DMA. Making the prefetch stream wait on the
        swap stream BEFORE entering the gather context is the minimal
        sufficient barrier: every chunk gather here waits for any
        in-flight SWAP D2H to retire on its source-side reads of the
        pool buffers. The wait is a single CUDA event — cheap relative
        to the H2D transfer it gates, and only added once per
        :meth:`pre_block_forward` / :meth:`pre_block_backward` call.

        This is the load-bearing primitive that makes ``BlockMode.SWAP``
        admissible on blocks whose parameter chunks are NOT in the
        persistent prefix (formerly rejected by
        :func:`block_map_runtime_admissible`). The pinned-CPU SWAP pool
        is the activation persistence mechanism — fully independent of
        ``param.data`` rebinding — so once the slot's D2H read is
        sequenced before any subsequent slot-overwriting H2D, the
        SWAP × non-persistent combination is byte-safe.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            return

        if self._prefetch_stream is None or not torch.cuda.is_available():
            # Synchronous fallback.
            for cid in chunk_ids:
                self.chunk_manager.gather(cid)
            return

        # Order this prefetch's H2D writes after any in-flight SWAP D2H
        # reads on the same pool buffers. See the docstring for why this
        # barrier is correctness-load-bearing for SWAP × non-persistent.
        # ``wait_stream`` records a single event and gates on it; no
        # host-side stall, and the cost on the GPU is dominated by the
        # H2D itself.
        if self._swap_stream is not None:
            self._prefetch_stream.wait_stream(self._swap_stream)

        with torch.cuda.stream(self._prefetch_stream):
            for cid in chunk_ids:
                # gather issues its own H2D copy with non_blocking=True; it
                # lands on the current stream (our prefetch stream).
                self.chunk_manager.gather(cid)

    def _sync_prefetch_with_compute(self) -> None:
        """Make the default compute stream wait on the prefetch stream."""
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if self._prefetch_stream is None or not torch.cuda.is_available():
            return
        compute = torch.cuda.current_stream()
        compute.wait_stream(self._prefetch_stream)

    def ensure_block_resident(self, block_id: BlockId) -> None:
        """Synchronously ensure ``block_id``'s parameter chunks are resident.

        Used by checkpoint recompute. ``torch.utils.checkpoint`` replays
        the inner block forward directly during backward, bypassing the
        wrapper module's forward-pre hook. The replay therefore needs a
        direct, idempotent gather hook before it touches the inner
        block's parameters.
        """
        chunk_ids = self._chunks_for(block_id)
        if not chunk_ids:
            return
        self._gather_on_prefetch_stream(chunk_ids)
        self._sync_prefetch_with_compute()

    # ---- forward -------------------------------------------------------

    def pre_block_forward(self, block_id: BlockId) -> None:
        """Prefetch the *next* block's chunks so they are resident by then.

        The **current** block's chunks are assumed to already be resident
        — they were either (a) kicked off by the previous block's
        ``pre_block_forward`` prefetch, or (b) persistent. On the very
        first block we also have to gather its own chunks, which we
        handle synchronously here to keep correctness.
        """
        # First-block warm-up: make sure the current block's chunks are in.
        # ``gather`` is idempotent on persistent chunks and fast on
        # already-resident non-persistent ones (it's just a tag lookup
        # through the pool). So calling unconditionally costs nothing in
        # steady state.
        self.ensure_block_resident(block_id)

        # Kick off async prefetch for the *next* block.
        nxt = self._next_block_of(block_id)
        if nxt is None:
            return
        next_chunks = self._chunks_for(nxt)
        if not next_chunks:
            return
        self._gather_on_prefetch_stream(next_chunks)
        # Do NOT sync here — the point of the prefetch stream is that
        # the copy can run overlapped with this block's forward compute.
        LOG.debug(
            "Scheduler.pre_block_forward: block=%d prefetched %d chunks for next block %d",
            block_id,
            len(next_chunks),
            nxt,
        )

    def post_block_forward(self, block_id: BlockId) -> None:
        """Release buffers whose last forward use was this block.

        Heuristic: release every non-persistent chunk owned by
        ``block_id`` *except* any that also appear in the next block's
        chunk set — keeping them resident lets the next block skip a
        re-gather on its pre-hook.

        The buffer pool preserves the chunk's tag after ``release`` so
        ``acquire_if_resident`` in backward still works (forward→backward
        reuse window, §3.1.1 + §5).
        """
        nxt = self._next_block_of(block_id)
        next_chunks: set[ChunkId] = (
            set(self._chunks_for(nxt)) if nxt is not None else set()
        )

        for cid in self._chunks_for(block_id):
            if cid in next_chunks:
                continue
            # ``offload`` short-circuits for persistent chunks — see
            # ChunkManager.offload docstring.
            self.chunk_manager.offload(cid)

    # ---- backward ------------------------------------------------------

    def pre_block_backward(self, block_id: BlockId) -> None:
        """Ensure the chunks for ``block_id`` are resident before its backward runs.

        Backward walks blocks in reverse order. The SWAP wrapper takes
        care of activation prefetch itself (`SwappedBlock`'s autograd
        Function schedules the H2D on the scheduler's ``_swap_stream``
        and synchronises the compute stream against it). We only need
        to cover the chunk-state path here.

        Fast path: if the chunk is still tagged in the buffer pool
        (``acquire_if_resident`` returns non-None) the gather call is a
        cheap re-tag + no-copy return. Otherwise the chunk manager
        re-gathers from the CPU shard with a fresh H2D copy.

        Lookahead: the chunk-prefetch lookahead at the bottom of this
        method already covers parameter chunks for block N-1 (the next
        backward block). For activation H2D the lookahead is implicit
        in the autograd graph — when block N's backward runs its
        ``_SwapOffloadFunction.backward``, the H2D for block N's
        activation lands on ``_swap_stream`` and the compute stream
        wait happens before block N's gradient kernels run. Block
        N-1's activation H2D will fire when *its* backward Function
        executes; the swap pool's ``prefetch_depth=2`` slots ensure
        block N's slot can be in-flight while block N-1's is being
        scheduled, mirroring the chunk-prefetch single-block
        lookahead pattern.
        """
        mode = self.block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.SWAP:
            LOG.debug(
                "Scheduler.pre_block_backward: block=%d is SWAP; "
                "activation H2D scheduled by SwappedBlock on swap_stream",
                block_id,
            )
        elif mode is BlockMode.OFFLOAD:
            # OFFLOAD-mode block: the wrapper installed
            # saved_tensors_hooks during forward; backward will fire an
            # unpack hook per saved param view that calls
            # ``ChunkManager.gather_for_backward(chunk_id)``. The
            # gather we issue below pre-warms the chunk so the unpack
            # hook hits the resident fast-path instead of forcing a
            # synchronous gather inside the autograd engine — see §3.3
            # of BLOCK_MODE_OFFLOAD_DESIGN. Ordering invariant: this
            # method runs from a backward-pre hook on the wrapper
            # module, which fires BEFORE autograd starts decoding the
            # block's saved tensors; that is what guarantees the
            # gather completes before the first unpack callback.
            LOG.debug(
                "Scheduler.pre_block_backward: block=%d is OFFLOAD; "
                "pre-warming chunk for saved-tensor unpack hook",
                block_id,
            )

        chunk_ids = self._chunks_for(block_id)
        if not chunk_ids:
            return

        # All-persistent layouts (n_buffer=0) skip pool construction
        # entirely — every chunk is GPU-resident throughout forward AND
        # backward, no gather/prefetch is needed here. The pool-cache
        # fast-path below would NPE on the missing pool; bail out
        # cleanly instead.
        if self.chunk_manager.buffer_pool is None:
            return

        # CRITICAL: a resident tag only proves a slot is *assigned* to
        # this chunk; the H2D copy that fills it may still be in flight
        # on ``_prefetch_stream`` (kicked off by the previous backward
        # step's lookahead at the bottom of this method, which
        # intentionally does NOT sync — see below). Compute reads on the
        # current stream must wait on that prefetch before trusting any
        # resident-tag hit, otherwise a "skip prefetch" decision races
        # the in-flight bytes and the gradient kernels see partially
        # populated memory. ``_sync_prefetch_with_compute`` is a
        # ``compute.wait_stream(prefetch_stream)`` — cheap when the
        # prefetch is already done, correct when it isn't.
        self._sync_prefetch_with_compute()

        # Consult the pool first — gathers that hit the resident tag are
        # essentially free; gathers that miss trigger a fresh H2D copy
        # onto the prefetch stream.
        misses: list[ChunkId] = []
        for cid in chunk_ids:
            if self.chunk_manager.buffer_pool.lookup_resident(cid) is None:
                misses.append(cid)
            else:
                # Re-claim the slot (removes from free list if present).
                self.chunk_manager.gather(cid)
        if misses:
            self._gather_on_prefetch_stream(misses)
            self._sync_prefetch_with_compute()

        # Also kick off an async prefetch for the block that is about to
        # be visited in the *next* backward step (i.e. the previous
        # block in forward order), mirroring the forward look-ahead.
        nxt_bwd = self._prev_block_of(block_id)
        if nxt_bwd is None:
            return
        nxt_chunks = self._chunks_for(nxt_bwd)
        if not nxt_chunks:
            return
        # Only gather what's not already resident to avoid needless work.
        need = [
            cid
            for cid in nxt_chunks
            if self.chunk_manager.buffer_pool.lookup_resident(cid) is None
        ]
        if need:
            self._gather_on_prefetch_stream(need)

    def post_block_backward(self, block_id: BlockId) -> None:
        """Finalize this block's backward: release buffers + maybe kick CPU Adam.

        Behavior after the M4.5 runtime-primitives landing:

        * **Non-persistent chunks** — grads for their params were already
          drained to the pinned-CPU grad shards by the per-parameter
          post-accumulate-grad hooks installed by
          :meth:`ChunkManager.materialize_offload` (the block-level hook
          used to own this, but could only fire after PyTorch's autograd
          had already accumulated grads for the whole block — too late
          for the memory-pressure path). The CPU FusedAdam step is
          kicked off inside those per-param hooks as soon as the last
          grad for a chunk lands. Here we merely release the GPU buffer
          and null ``param.data`` so the slot can be recycled.
        * **Persistent chunks** — their grads live on GPU (no drain);
          the call is a no-op in single-rank mode, and in multi-rank
          mode issues the distributed all-reduce per param.
        """
        for cid in self._chunks_for(block_id):
            # Block-contiguity rule (§3.1.1): a chunk can be shared with an
            # adjacent block. Only the EARLIEST forward-order owner — i.e.
            # the LAST block to visit the chunk in backward — should
            # finalize it. Skipping here lets the earlier block's
            # post_block_backward fire the reduce-and-offload once all
            # owners have produced their grads. See the
            # ``_chunk_last_bwd_owner`` precomputation in ``__init__``.
            if self._chunk_last_bwd_owner.get(cid, block_id) != block_id:
                continue
            self.chunk_manager.reduce_grads_and_offload(cid)

    # ---- end-of-iteration cleanup -------------------------------------

    def drain(self) -> None:
        """Block until every in-flight CPU Adam step has finished.

        Called at the end of ``backward`` (or at the start of the next
        ``optimizer.step``) so the non-persistent optimizer updates are
        committed before the next forward observes stale params.

        OFFLOAD-mode integration (M3, §3.3 of
        BLOCK_MODE_OFFLOAD_DESIGN): we also drain any chunks whose
        offload was deferred because a ``BackwardHandle`` was
        outstanding at ``reduce_grads_and_offload`` time. In steady
        state ``BackwardHandle.__del__`` already drains via Python
        ref-counting on the unpack-returned view, so the drain call
        here is defensive — it makes the timing explicit, composable
        with future schedulers, and assertable by debug paths.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            # CPU-only path: still flush deferred offloads so the
            # contract holds even without CUDA available.
            self.chunk_manager.drain_deferred_offloads()
            self.chunk_manager.wait_cpu_optim()
            return

        # Make sure any prefetch / swap traffic that's still inflight
        # completes before we declare the iteration done — callers
        # inspecting peak memory stats right after drain expect a stable
        # picture.
        if torch.cuda.is_available():
            if self._prefetch_stream is not None:
                self._prefetch_stream.synchronize()
            if self._swap_stream is not None:
                self._swap_stream.synchronize()

        # Defensive end-of-iter drain for OFFLOAD-mode chunks. Any
        # chunk whose backward refcount is still > 0 here indicates an
        # autograd-engine reference that hasn't been released — leave
        # it queued and the eventual handle drop will offload it.
        # ``drain_deferred_offloads`` only runs on refcount==0 entries.
        self.chunk_manager.drain_deferred_offloads()

        self.chunk_manager.wait_cpu_optim()


__all__ = ["Scheduler"]
