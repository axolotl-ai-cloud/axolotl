"""Pre-allocated GPU chunk buffer pool with oversize-chunk side-table."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque

from axolotl.integrations.protrain.types import ChunkId
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

LOG = get_logger(__name__)


class BufferPool:
    """Fixed pool of GPU chunk buffers with forward→backward reuse tracking.

    LRU-free eviction: prefer the buffer least-recently released
    so the most-recently-used chunks stay resident longest. We implement
    this with a FIFO of free slots where ``release`` appends and ``acquire``
    pops the oldest — standard LRU.

    Dtype notes (M4.5)
    ------------------
    Buffers are allocated as flat uint8 GPU tensors. The
    :class:`ChunkManager` reinterprets each buffer on gather via
    ``buf.narrow(0, offset, nbytes).view(dtype).view(shape)`` per param
    slot, matching the layout built by
    :meth:`ChunkManager.materialize_offload`. This keeps the pool dtype-
    agnostic (works for mixed-dtype chunks — e.g. fp16 weights and fp32
    lm_head tied-weight cases) at the cost of storing the per-param
    ``(offset, dtype, shape)`` metadata on the ChunkManager's
    ``_cpu_slots`` table rather than in the pool.
    """

    def __init__(
        self,
        n_buffer: int,
        S_chunk: int,
        pinned_host: "PinnedHostMemory",
        device: "torch.device | str",
    ) -> None:
        """Pre-allocate ``n_buffer`` flat ``S_chunk``-byte GPU buffers and the free list."""
        if n_buffer <= 0:
            raise ValueError(f"n_buffer must be positive, got {n_buffer}")
        if S_chunk <= 0:
            raise ValueError(f"S_chunk must be positive, got {S_chunk}")
        if pinned_host.n_buffer != n_buffer or pinned_host.S_chunk != S_chunk:
            raise ValueError(
                f"pinned_host shape ({pinned_host.n_buffer}x{pinned_host.S_chunk}) "
                f"must match pool ({n_buffer}x{S_chunk})"
            )

        # Local import so the module can be imported without torch present.
        import torch

        self.n_buffer = int(n_buffer)
        self.S_chunk = int(S_chunk)
        self.pinned_host = pinned_host
        self.device = torch.device(device)

        # Pre-allocate via SingleStreamAllocator so pool slots land on default-stream heap.
        if self.device.type == "cuda" and torch.cuda.is_available():
            from axolotl.integrations.protrain.runtime.streams import (
                SingleStreamAllocator,
            )

            with SingleStreamAllocator():
                self._buffers: list["torch.Tensor"] = [
                    torch.empty(self.S_chunk, dtype=torch.uint8, device=self.device)
                    for _ in range(self.n_buffer)
                ]
        else:
            # CPU-only path (test lanes without CUDA): no heap concept, no
            # need for the allocator context — a plain construction keeps
            # the import path light.
            self._buffers = [
                torch.empty(self.S_chunk, dtype=torch.uint8, device=self.device)
                for _ in range(self.n_buffer)
            ]
        # Tag survives release for forward→backward reuse lookup.
        self._tags: list[ChunkId | None] = [None] * self.n_buffer
        # FIFO free list = LRU with release-on-use; _free_set is O(1) membership.
        self._free: Deque[int] = deque(range(self.n_buffer))
        self._free_set: set[int] = set(range(self.n_buffer))
        # Reverse map for O(1) resident lookup.
        self._tag_to_slot: dict[ChunkId, int] = {}
        # Per-slot lease refcount; cache-hit reuse requires this for safe two-caller release.
        self._leases: list[int] = [0] * self.n_buffer

        # Oversize-chunk side-table for chunks > S_chunk; own lease counters.
        self._large_buffers: dict[ChunkId, "torch.Tensor"] = {}
        self._large_leases: dict[ChunkId, int] = {}

        # close() flag for idempotent teardown.
        self._closed: bool = False

    # ---- core ops ------------------------------------------------------

    def acquire(
        self, chunk_id: ChunkId, chunk_bytes: int | None = None
    ) -> "torch.Tensor":
        """Return buffer for chunk_id (cache hit) or evict LRU free slot (miss)."""
        if self._closed:
            raise RuntimeError(
                "BufferPool.acquire: pool is closed; cannot acquire "
                f"chunk {chunk_id}. This indicates a lifecycle bug — "
                "the pool was torn down (close() called) before the "
                "caller stopped issuing acquires."
            )
        # Oversize fast path: side-table is disjoint from slot state.
        if chunk_bytes is not None and chunk_bytes > self.S_chunk:
            existing = self._large_buffers.get(chunk_id)
            if existing is not None:
                # Two-prefetch convergence; bump lease and return same tensor.
                if existing.numel() != chunk_bytes:
                    raise RuntimeError(
                        f"BufferPool: oversize buffer for chunk {chunk_id} "
                        f"already allocated at {existing.numel()} bytes; "
                        f"caller requested {chunk_bytes} bytes"
                    )
                self._large_leases[chunk_id] = self._large_leases.get(chunk_id, 0) + 1
                return existing
            # First-time alloc; SingleStreamAllocator → default-stream heap.
            import torch

            if self.device.type == "cuda" and torch.cuda.is_available():
                from axolotl.integrations.protrain.runtime.streams import (
                    SingleStreamAllocator,
                )

                with SingleStreamAllocator():
                    buf = torch.empty(
                        chunk_bytes, dtype=torch.uint8, device=self.device
                    )
            else:
                buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=self.device)
            self._large_buffers[chunk_id] = buf
            self._large_leases[chunk_id] = 1
            return buf

        # Cache hit: chunk already in a slot.
        slot = self._tag_to_slot.get(chunk_id)
        if slot is not None:
            # Discard from free set + remove from deque eagerly; n_buffer is small.
            self._free_set.discard(slot)
            try:
                self._free.remove(slot)
            except ValueError:
                pass
            self._leases[slot] += 1
            return self._buffers[slot]

        if not self._free_set:
            raise RuntimeError(
                f"BufferPool exhausted: all {self.n_buffer} buffers in use, "
                f"cannot acquire for chunk {chunk_id}. Increase n_buffer "
                "or release buffers before acquiring new ones."
            )

        # Pop oldest entry still in _free_set; skip stale deque entries.
        while True:
            slot = self._free.popleft()
            if slot in self._free_set:
                self._free_set.discard(slot)
                break
        # Evict the previous tag's mapping.
        prev_tag = self._tags[slot]
        if prev_tag is not None:
            self._tag_to_slot.pop(prev_tag, None)
        self._tags[slot] = chunk_id
        self._tag_to_slot[chunk_id] = slot
        # Set (not increment) since the slot came off free with lease=0.
        self._leases[slot] = 1
        return self._buffers[slot]

    def release(self, chunk_id: ChunkId) -> None:
        """Return chunk_id's buffer to the free list (preserves tag); oversize entries dropped."""
        if self._closed:
            return
        # Oversize: free eagerly when lease drops to zero; disjoint from slot state.
        if chunk_id in self._large_buffers:
            count = self._large_leases.get(chunk_id, 0)
            if count <= 1:
                self._large_buffers.pop(chunk_id, None)
                self._large_leases.pop(chunk_id, None)
            else:
                self._large_leases[chunk_id] = count - 1
            return
        slot = self._tag_to_slot.get(chunk_id)
        if slot is None:
            return
        if self._leases[slot] == 0:
            return  # double-release no-op
        self._leases[slot] -= 1
        if self._leases[slot] > 0:
            return
        # Append (LRU-free); _free_set mirrored for O(1) discard.
        self._free.append(slot)
        self._free_set.add(slot)

    def lookup_resident(self, chunk_id: ChunkId) -> "torch.Tensor | None":
        """Lease-free peek; returned buffer is only safe until next acquire().

        Prefer :meth:`acquire_if_resident` for any caller that intends to
        actually read the buffer — it takes a real lease and pairs with
        ``release(chunk_id)``, eliminating the eviction race while
        preserving the "no eviction on miss" semantics.

        Oversize chunks: a present entry in :attr:`_large_buffers` is
        always safe to read — there's no concurrent eviction path
        because oversize buffers are dropped only at :meth:`release`.

        After :meth:`close`, returns ``None`` — there is no resident
        state left to peek at. This is the safe, non-raising
        counterpart to the lease-taking :meth:`acquire_if_resident`
        guard: lookups can legitimately race against teardown, so
        returning ``None`` (rather than raising) lets callers fall
        through their miss path.
        """
        if self._closed:
            return None
        large = self._large_buffers.get(chunk_id)
        if large is not None:
            return large
        slot = self._tag_to_slot.get(chunk_id)
        if slot is None:
            return None
        return self._buffers[slot]

    def acquire_if_resident(self, chunk_id: ChunkId) -> "torch.Tensor | None":
        """Lease-taking variant of :meth:`lookup_resident`.

        On a tag hit, behaves exactly like :meth:`acquire` (increments
        the slot's lease, removes the slot from the free list if present)
        and returns the buffer. On a miss returns ``None`` *without*
        evicting any other slot — the caller can then decide between a
        full ``acquire`` (which evicts) or a different recovery path.

        Pair every successful return with a matching ``release(chunk_id)``
        once the caller is done reading the buffer; otherwise the slot
        stays leased and cannot be recycled.

        Use this instead of :meth:`lookup_resident` whenever you intend
        to actually read the buffer's bytes — it closes the
        peek-then-evict race window where another ``acquire`` between
        the lookup and the read could re-tag the slot and overwrite the
        data.

        Oversize chunks: a hit in :attr:`_large_buffers` increments
        the side-table lease (mirroring the slot-pool path at line
        425) so the matching ``release(chunk_id)`` only drops the
        buffer when the LAST holder releases.

        Raises ``RuntimeError`` if the pool has been closed — mirrors
        :meth:`acquire` so post-close residency probes surface the
        lifecycle bug instead of silently returning stale state.
        """
        if self._closed:
            raise RuntimeError(
                "BufferPool.acquire_if_resident: pool is closed; cannot "
                f"acquire chunk {chunk_id}. This indicates a lifecycle "
                "bug — the pool was torn down (close() called) before "
                "the caller stopped issuing residency probes."
            )
        large = self._large_buffers.get(chunk_id)
        if large is not None:
            self._large_leases[chunk_id] = self._large_leases.get(chunk_id, 0) + 1
            return large
        slot = self._tag_to_slot.get(chunk_id)
        if slot is None:
            return None
        # Same lease-bookkeeping as the cache-hit fast path in acquire().
        self._free_set.discard(slot)
        try:
            self._free.remove(slot)
        except ValueError:
            pass
        self._leases[slot] += 1
        return self._buffers[slot]

    def invalidate_tag(self, chunk_id: ChunkId) -> None:
        """Drop residency tag for chunk_id; raises if slot is leased."""
        slot = self._tag_to_slot.get(chunk_id)
        if slot is None:
            return
        if self._leases[slot] > 0:
            raise RuntimeError(
                f"BufferPool.invalidate_tag: cannot invalidate "
                f"chunk_id={chunk_id} while slot {slot} has "
                f"{self._leases[slot]} active lease(s). The caller "
                "must release all leases before invalidating; otherwise "
                "the matching release() would silently no-op and leave "
                "the slot permanently in-use."
            )
        self._tag_to_slot.pop(chunk_id, None)
        self._tags[slot] = None

    # ---- introspection -------------------------------------------------

    @property
    def num_free(self) -> int:
        # _free_set is authoritative; _free can carry stale entries.
        return len(self._free_set)

    @property
    def num_in_use(self) -> int:
        return self.n_buffer - self.num_free

    def __len__(self) -> int:
        return self.n_buffer

    # ---- teardown ------------------------------------------------------

    def close(self) -> None:
        """Drop pool buffers + paired pinned region (host first so teardown is retryable)."""
        if self._closed:
            return
        # Release pinned host first; on raise leave pool retryable instead of leaking the pinned allocation.
        if self.pinned_host is not None:
            self.pinned_host.close()
            self.pinned_host = None  # type: ignore[assignment]
        self._closed = True
        self._buffers = []
        self._large_buffers.clear()
        self._large_leases.clear()
        self._free.clear()
        self._free_set.clear()
        self._tag_to_slot.clear()
        self._tags = [None] * self.n_buffer
        self._leases = [0] * self.n_buffer


__all__ = ["BufferPool"]
