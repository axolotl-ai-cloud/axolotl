"""Block-granularity runtime scheduler: prefetch/release/reduce-offload at block boundaries."""

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

        self._prefetch_stream: "torch.cuda.Stream | None" = None
        self._swap_stream: "torch.cuda.Stream | None" = None
        # Lazy ActivationSwapPool attach by wrapper when n_swap > 0.
        self.swap_pool: object | None = None
        self._closed: bool = False
        self._init_streams()

    @property
    def swap_stream(self) -> "torch.cuda.Stream | None":
        """Public accessor for the dedicated activation-swap stream (None on CPU)."""
        return self._swap_stream

    def _init_streams(self) -> None:
        """Create dedicated CUDA streams for prefetch + activation swap."""
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

        # Non-default stream lets compute overlap PCIe copies.
        self._prefetch_stream = torch.cuda.Stream()
        # Separate swap stream avoids contention with chunk prefetch.
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
        """Return the next-in-backward block (idx-1 in forward order)."""
        idx = self._block_index_map.get(block_id)
        if idx is None or idx <= 0:
            return None
        return self._block_order[idx - 1]

    def _gather_on_prefetch_stream(self, chunk_ids: Iterable[ChunkId]) -> None:
        """Async-gather chunk_ids on the prefetch stream; waits on swap_stream to sequence SWAP D2H."""
        try:
            import torch
        except ImportError:  # pragma: no cover
            return

        if self._prefetch_stream is None or not torch.cuda.is_available():
            # Synchronous fallback.
            for cid in chunk_ids:
                self.chunk_manager.gather(cid)
            return

        # Order H2D writes after in-flight SWAP D2H reads (correctness for SWAP × non-persistent).
        if self._swap_stream is not None:
            self._prefetch_stream.wait_stream(self._swap_stream)

        with torch.cuda.stream(self._prefetch_stream):
            for cid in chunk_ids:
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
        """Sync gather block's chunks; used by checkpoint recompute path that bypasses pre-hooks."""
        chunk_ids = self._chunks_for(block_id)
        if not chunk_ids:
            return
        self._gather_on_prefetch_stream(chunk_ids)
        self._sync_prefetch_with_compute()

    def ensure_chunks_resident(self, chunk_ids: Iterable[ChunkId]) -> None:
        """Sync gather chunks on compute stream so autograd sees real param.size() on cold paths."""
        cids = tuple(chunk_ids)
        if not cids:
            return
        # Wait on swap+prefetch so pool buffers and in-flight gathers complete pre-rebind.
        try:
            import torch as _torch
        except ImportError:  # pragma: no cover — defensive, CPU-only lanes
            _torch = None  # type: ignore[assignment]
        if _torch is not None and _torch.cuda.is_available():
            compute = _torch.cuda.current_stream()
            if self._swap_stream is not None:
                compute.wait_stream(self._swap_stream)
            if self._prefetch_stream is not None:
                compute.wait_stream(self._prefetch_stream)
        # gather on compute stream so sharded all_gather completes before autograd records source-shape.
        for cid in cids:
            self.chunk_manager.gather(cid)

    # ---- forward -------------------------------------------------------

    def pre_block_forward(self, block_id: BlockId) -> None:
        """Prefetch the next block's chunks; ensure current block's are resident."""
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

    def post_block_forward(self, block_id: BlockId) -> None:
        """Release this block's non-persistent chunks except those used by the next block."""
        nxt = self._next_block_of(block_id)
        next_chunks: set[ChunkId] = (
            set(self._chunks_for(nxt)) if nxt is not None else set()
        )

        for cid in self._chunks_for(block_id):
            if cid in next_chunks:
                continue
            # offload() short-circuits for persistent chunks.
            self.chunk_manager.offload(cid)

    # ---- backward ------------------------------------------------------

    def pre_block_backward(self, block_id: BlockId) -> None:
        """Ensure block's chunks are resident before backward; cover chunk-state path only (SWAP handles activations)."""
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

        # Consult pool: hits are free, misses trigger fresh H2D on prefetch stream.
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
            self._gather_on_prefetch_stream(need)

    def post_block_backward(self, block_id: BlockId) -> None:
        """Finalize block's backward: release buffers + maybe kick CPU Adam."""
        for cid in self._chunks_for(block_id):
            # Shared chunks: only earliest-forward owner finalizes.
            if self._chunk_last_bwd_owner.get(cid, block_id) != block_id:
                continue
            self.chunk_manager.reduce_grads_and_offload(cid)

    # ---- end-of-iteration cleanup -------------------------------------

    def drain(self) -> None:
        """Block until every in-flight CPU Adam step has finished; flush deferred offloads."""
        try:
            import torch
        except ImportError:  # pragma: no cover
            # CPU-only path: still flush deferred offloads.
            self.chunk_manager.drain_deferred_offloads()
            self.chunk_manager.wait_cpu_optim()
            return

        # Drain in-flight prefetch/swap traffic for stable peak-memory stats.
        if torch.cuda.is_available():
            if self._prefetch_stream is not None:
                self._prefetch_stream.synchronize()
            if self._swap_stream is not None:
                self._swap_stream.synchronize()

        # Defensive end-of-iter drain (only refcount==0 entries fire).
        self.chunk_manager.drain_deferred_offloads()

        self.chunk_manager.wait_cpu_optim()

    # ---- teardown -----------------------------------------------------

    def close(self) -> None:
        """Synchronize + drop streams and swap pool. Idempotent. Does NOT close chunk manager."""
        if self._closed:
            return
        self._closed = True
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None  # type: ignore[assignment]
        if torch is not None and torch.cuda.is_available():
            for stream in (self._prefetch_stream, self._swap_stream):
                if stream is None:
                    continue
                try:
                    stream.synchronize()
                except Exception as exc:  # noqa: BLE001 — best-effort
                    LOG.debug("Scheduler.close: stream synchronize failed: %s", exc)
        self._prefetch_stream = None
        self._swap_stream = None
        if self.swap_pool is not None:
            try:
                self.swap_pool.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("Scheduler.close: swap_pool.close failed: %s", exc)
            self.swap_pool = None


__all__ = ["Scheduler"]
