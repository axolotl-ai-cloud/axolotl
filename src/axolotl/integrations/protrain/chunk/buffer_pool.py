"""Pre-allocated GPU chunk buffer pool.

A fixed pool of ``n_buffer`` GPU tensors of ``S_chunk`` bytes each. Every
non-persistent chunk gather borrows a buffer; ``release`` returns it. Buffers
carry a ``chunk_id`` tag so the backward pass can ask "is this chunk's data
still resident in one of my buffers?" via :meth:`lookup_resident` — if yes,
we skip the reload. §3.1.1 + §5.

Paired with :class:`~axolotl.integrations.protrain.chunk.pinned_alloc.PinnedHostMemory`
for the host-side staging region of the same shape.
"""

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

    The pool owns ``n_buffer`` GPU ``uint8`` tensors, each exactly
    ``S_chunk`` bytes. Callers reinterpret them via ``.view(dtype)`` as
    needed. A paired :class:`PinnedHostMemory` provides the CPU-side staging
    slots (same index space), so H2D copies are pinned→device and hit peak
    PCIe throughput.

    Semantics:

    * :meth:`acquire(chunk_id)` — take a free buffer and tag it with the
      chunk. If the chunk is already resident (tag match), return the same
      buffer (reuse path from forward into backward).
    * :meth:`release(chunk_id)` — return the buffer to the free list. The
      tag is *preserved* so a subsequent :meth:`lookup_resident` still sees
      it; the buffer is only actually overwritten when it's re-acquired
      for a different chunk, at which point its tag is updated.
    * :meth:`lookup_resident(chunk_id)` — ``None`` unless a buffer with a
      matching tag exists; returns the buffer regardless of whether it's
      currently in the free list (the backward pass uses this to skip
      redundant H2D copies).

    The "LRU-free" wording in the spec means: when multiple buffers are
    free and we must evict one, prefer the buffer least-recently released
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

        # Pre-allocate every buffer up-front — the whole point of the pool
        # is to avoid allocator churn during training. Route through the
        # default-stream allocator (paper App B.2 / SingleStreamAllocator
        # contract) so every long-lived pool slot sits on the default
        # stream's heap. These slots are the largest sustained GPU
        # allocation in ProTrain (n_buffer × S_chunk bytes — tens of
        # gigabytes on training-scale models), so unifying their heap
        # is the highest-leverage single application of App B.2. No
        # ``record_stream`` needed: the slots' lifetime is owned by the
        # pool and they only return to the allocator at pool teardown,
        # by which point every consuming stream has drained.
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
        # Per-slot chunk tag; ``None`` means "never held a chunk". This
        # tag survives ``release`` so the forward→backward reuse lookup
        # works even after a buffer has been handed back to the free list.
        self._tags: list[ChunkId | None] = [None] * self.n_buffer
        # FIFO free list → effectively LRU when combined with release-on-use.
        self._free: Deque[int] = deque(range(self.n_buffer))
        # O(1) free-membership tracker paired with the deque. The deque
        # preserves LRU ordering for popleft-eviction; ``_free_set`` lets
        # us check / remove by slot id without scanning the deque (the
        # cache-hit path's ``self._free.remove(slot)`` was O(n)). The
        # deque can carry stale entries — they're filtered lazily on
        # popleft via the membership check below.
        self._free_set: set[int] = set(range(self.n_buffer))
        # Reverse map for O(1) resident lookup.
        self._tag_to_slot: dict[ChunkId, int] = {}
        # Per-slot lease refcount. ``acquire`` increments (cache hit) or sets
        # to 1 (miss); ``release`` decrements and only returns the slot to
        # ``_free`` when the count hits 0. Without this, a cache-hit handing
        # the same buffer to two callers would let the first ``release`` put
        # the slot back on the free list while the second caller still holds
        # it, allowing a subsequent miss to overwrite live data.
        self._leases: list[int] = [0] * self.n_buffer

    # ---- core ops ------------------------------------------------------

    def acquire(self, chunk_id: ChunkId) -> "torch.Tensor":
        """Return a buffer holding ``chunk_id``; allocate from the free list if needed.

        If the chunk is already resident and its slot is in the free list,
        we re-claim the same slot (no H2D copy needed at the call site).
        If the chunk isn't resident we evict the LRU free slot, re-tag it
        with ``chunk_id``, and return it (the caller is responsible for the
        H2D copy that follows).
        """
        # Fast path: chunk is already in a slot (possibly free, possibly in-use).
        slot = self._tag_to_slot.get(chunk_id)
        if slot is not None:
            # O(1) free-set discard — the deque may still carry the
            # stale entry, but ``popleft`` below filters via the set.
            self._free_set.discard(slot)
            self._leases[slot] += 1
            return self._buffers[slot]

        if not self._free_set:
            raise RuntimeError(
                f"BufferPool exhausted: all {self.n_buffer} buffers in use, "
                f"cannot acquire for chunk {chunk_id}. Increase n_buffer "
                "or release buffers before acquiring new ones."
            )

        # Pop the oldest entry that's still in ``_free_set``. Stale
        # entries (slots claimed by the cache-hit fast path above without
        # the matching deque-rewrite) are skipped here in O(1) amortized.
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
        # Freshly allocated to this chunk_id — set (don't increment) since
        # the slot just came off the free list with a 0 lease count.
        self._leases[slot] = 1
        return self._buffers[slot]

    def release(self, chunk_id: ChunkId) -> None:
        """Return ``chunk_id``'s buffer to the free list, preserving its tag.

        Silently no-op if the chunk isn't currently held — callers can
        release unconditionally without special-casing the persistent path.
        """
        slot = self._tag_to_slot.get(chunk_id)
        if slot is None:
            return
        if self._leases[slot] == 0:
            return  # already released (double-release is a safe no-op)
        self._leases[slot] -= 1
        if self._leases[slot] > 0:
            # Still leased by another caller (cache-hit reuse path) — the
            # slot must stay off the free list until the last lease drops.
            return
        # Append (not appendleft) to implement LRU-free: the oldest free
        # slot gets evicted first on the next ``acquire`` that misses.
        # Mirror in ``_free_set`` so the cache-hit fast path can do an
        # O(1) discard.
        self._free.append(slot)
        self._free_set.add(slot)

    def lookup_resident(self, chunk_id: ChunkId) -> "torch.Tensor | None":
        """Return the buffer if the chunk's data is still tagged in a slot.

        Used by the backward pass to detect that forward's buffer was never
        evicted — in which case no H2D re-gather is needed. Returns ``None``
        if the tag has been overwritten by an intervening ``acquire``.

        Lease semantics — IMPORTANT
        ---------------------------
        This is a *peek*: it does NOT take a lease on the slot. The returned
        buffer pointer is only safe to read so long as no other ``acquire``
        intervenes; if the slot happens to be in the free list (lease==0),
        the very next ``acquire()`` for a different chunk could evict the
        tag and overwrite the bytes.

        The current callers honour this by either:

        * immediately calling ``acquire(chunk_id)`` (or ``gather()``, which
          calls ``acquire`` internally) on the same chunk_id, taking a real
          lease before any other ``acquire`` can run on the same thread; or
        * already holding an independent refcount on the chunk via the
          chunk-manager's ``gather_for_backward`` / ``BackwardHandle``
          machinery, which prevents ``offload`` from being scheduled
          (and thus prevents ``acquire`` from claiming this slot for
          a different chunk) for the duration of the backward unpack.

        New callers MUST satisfy one of those two conditions, or use
        ``acquire(chunk_id)`` directly to take a real lease. Treating the
        return value as a long-lived borrow without one of those guarantees
        is a use-after-evict race.
        """
        slot = self._tag_to_slot.get(chunk_id)
        if slot is None:
            return None
        return self._buffers[slot]

    # ---- introspection -------------------------------------------------

    @property
    def num_free(self) -> int:
        # ``_free`` can carry stale entries (the lazy-cleanup path in
        # ``acquire``); ``_free_set`` is the authoritative count.
        return len(self._free_set)

    @property
    def num_in_use(self) -> int:
        return self.n_buffer - self.num_free

    def __len__(self) -> int:
        return self.n_buffer


__all__ = ["BufferPool"]
