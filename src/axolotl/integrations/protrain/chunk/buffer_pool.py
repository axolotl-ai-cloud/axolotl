"""Pre-allocated GPU chunk buffer pool.

A fixed pool of ``n_buffer`` GPU tensors of ``S_chunk`` bytes each. Every
non-persistent chunk gather borrows a buffer; ``release`` returns it. Buffers
carry a ``chunk_id`` tag so the backward pass can ask "is this chunk's data
still resident in one of my buffers?" via :meth:`acquire_if_resident` (the
lease-taking, race-safe API) — if yes, we skip the reload. §3.1.1 + §5.
:meth:`lookup_resident` remains as a lease-free peek for boolean residency
probes that don't read the buffer.

Paired with :class:`~axolotl.integrations.protrain.chunk.pinned_alloc.PinnedHostMemory`
for the host-side staging region of the same shape.

Oversize-chunk side-table (§3 + paper-fidelity addon)
-----------------------------------------------------
The paper's :math:`M_{buffer} = n_{buffer} \\times S_{chunk}` footprint
ceiling (Eq. 11) refers to the *slot pool* — ``n_buffer`` uniform-size
buffers that host normal chunks. The layout builder, however, supports
placing a single param larger than ``S_chunk`` in its own chunk
(``layout.py``: "A single param larger than ``S_chunk`` is placed on its
own in a fresh chunk"); the S_chunk picker accepts these (sizing.py
clamps oversize-chunk waste at 0 — it knows the chunks will be oversize
and is OK with that). To handle them at runtime without violating the
slot-uniform invariant, this pool keeps a separate
:attr:`_large_buffers` dict, keyed on ``ChunkId``, holding one-off
exact-byte allocations for chunks where ``chunk_bytes > S_chunk``.

Oversize allocations:

* Do NOT compete for slot leases — they don't pop from ``_free`` and
  don't increment ``_leases``. The slot pool's lease-counter invariants
  are unaffected.
* Pass through :class:`SingleStreamAllocator` like the slot pool itself
  so they land on the default-stream heap. This means the running
  reserved-bytes total can transiently exceed ``n_buffer * S_chunk`` by
  the sum of currently-resident oversize-chunk bytes; this is documented
  in the cost model as the OFFLOAD-mode runtime overhead. The
  alternative — allocating off the default heap with a ``record_stream``
  cleanup — would force every gather/release to drag a stream-event
  through the per-chunk control flow with no correctness benefit.
* Are released (dropped from the dict) on :meth:`release`, NOT preserved
  for forward→backward H2D-skip reuse. The slot-pool tag-preserving
  optimization buys an avoided H2D for normal chunks at the cost of
  keeping their bytes resident; oversize chunks would pay the same cost
  in proportionally more bytes, so we drop them eagerly. The scheduler's
  ``_active_chunks`` lease-idempotency contract (manager.py) handles
  duplicate gathers within a single active window, so the
  reuse-on-release hit is bounded.
* Are completely transparent when no oversize chunks exist — every code
  path through this module short-circuits when ``chunk_bytes <= S_chunk``
  (or is omitted), so models without oversize chunks behave identically
  to the pre-oversize implementation.
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
    * :meth:`acquire_if_resident(chunk_id)` — ``None`` on miss; on hit
      takes a lease and returns the leased buffer (race-safe against
      concurrent eviction). The backward pass uses this to skip
      redundant H2D copies. Pair with :meth:`release(chunk_id)`.
    * :meth:`lookup_resident(chunk_id)` — lease-free peek for boolean
      residency probes that don't dereference the returned buffer
      (e.g. routing decisions). Returning a buffer here does NOT
      protect against eviction; use :meth:`acquire_if_resident` if
      you need to read.

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
        # allocation in ProTrain (n_buffer x S_chunk bytes — tens of
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

        # Oversize-chunk side-table — see the module docstring's
        # "Oversize-chunk side-table" section. Each entry holds one
        # exact-byte uint8 GPU tensor for a chunk whose ``chunk_bytes``
        # exceeds ``self.S_chunk``. Empty in the common case (no
        # oversize chunks), which keeps the normal slot-pool path
        # zero-overhead. Each oversize buffer carries its own
        # lease counter (``_large_leases``) mirroring the slot-pool
        # ``_leases`` discipline: ``acquire``/``acquire_if_resident``
        # increment, ``release`` decrements, and the side-table entry
        # is dropped only when the lease drops to zero. This closes
        # the "two prefetch sites converged on the same chunk" race
        # where the first ``release`` would otherwise drop the buffer
        # while a second caller still references it.
        self._large_buffers: dict[ChunkId, "torch.Tensor"] = {}
        self._large_leases: dict[ChunkId, int] = {}

        # Set by :meth:`close`. Once flipped, all public ops are no-ops so
        # a teardown cascade (WrappedModel.close -> ChunkManager.close ->
        # BufferPool.close) is idempotent under repeated invocation.
        self._closed: bool = False

    # ---- core ops ------------------------------------------------------

    def acquire(
        self, chunk_id: ChunkId, chunk_bytes: int | None = None
    ) -> "torch.Tensor":
        """Return a buffer holding ``chunk_id``; allocate from the free list if needed.

        If the chunk is already resident and its slot is in the free list,
        we re-claim the same slot (no H2D copy needed at the call site).
        If the chunk isn't resident we evict the LRU free slot, re-tag it
        with ``chunk_id``, and return it (the caller is responsible for the
        H2D copy that follows).

        When ``chunk_bytes is not None and chunk_bytes > self.S_chunk``
        the chunk is "oversize": we route it to the per-chunk
        :attr:`_large_buffers` side-table instead of the slot pool. See
        the module docstring's "Oversize-chunk side-table" section for
        the full contract. Oversize allocations do NOT consume slot
        leases, so passing ``chunk_bytes`` with a tiny value is a no-op
        from the slot pool's perspective — the pool's
        ``M_buffer = n_buffer * S_chunk`` paper-Eq. 11 ceiling stays
        valid for the slot pool itself.

        Raises ``RuntimeError`` if the pool has been closed via
        :meth:`close`; surfacing the lifecycle error here keeps a stale
        re-wrap path from silently re-allocating buffers behind a
        torn-down pool.
        """
        if self._closed:
            raise RuntimeError(
                "BufferPool.acquire: pool is closed; cannot acquire "
                f"chunk {chunk_id}. This indicates a lifecycle bug — "
                "the pool was torn down (close() called) before the "
                "caller stopped issuing acquires."
            )
        # Oversize fast path: route through the side-table BEFORE
        # touching any slot state. This keeps the slot lease counter,
        # free-set, and tag table identical to a model with no oversize
        # chunks — the path here is invisible to the slot machinery.
        if chunk_bytes is not None and chunk_bytes > self.S_chunk:
            existing = self._large_buffers.get(chunk_id)
            if existing is not None:
                # Lease-idempotent within an active window: the
                # ChunkManager's ``_active_chunks`` set is what gates
                # duplicate gathers from re-entering this branch in
                # steady-state, so reaching here with an existing
                # entry is the legitimate "two prefetch sites
                # converged on the same chunk" case. Return the same
                # tensor; no double-allocation.
                if existing.numel() != chunk_bytes:
                    # Defensive — should never happen because chunk
                    # bytes are a function of layout + dtype which are
                    # immutable for a given manager lifetime. If a
                    # caller drifted, surface it loudly rather than
                    # silently returning a wrong-size buffer.
                    raise RuntimeError(
                        f"BufferPool: oversize buffer for chunk {chunk_id} "
                        f"already allocated at {existing.numel()} bytes; "
                        f"caller requested {chunk_bytes} bytes"
                    )
                # Increment the lease so a paired ``release(chunk_id)``
                # from each acquirer is required before the buffer is
                # actually dropped. Mirrors the slot-pool discipline
                # at line 294 (``self._leases[slot] += 1``).
                self._large_leases[chunk_id] = self._large_leases.get(chunk_id, 0) + 1
                return existing
            # First-time allocation for this oversize chunk. Route
            # through ``SingleStreamAllocator`` so the bytes land on
            # the default-stream heap — same App B.2 contract the
            # slot pool itself observes (constructor above). The
            # allocation can transiently push reserved bytes above
            # ``n_buffer * S_chunk`` by ``chunk_bytes`` worth of
            # bytes; this is the documented oversize cost.
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
            # Initialize the lease at 1 — same convention as the slot
            # pool (line 320: ``self._leases[slot] = 1``). The first
            # ``release(chunk_id)`` will decrement to 0 and drop the
            # buffer; additional ``acquire``/``acquire_if_resident``
            # calls within the same active window will bump the count
            # to keep the buffer alive until ALL holders release.
            self._large_leases[chunk_id] = 1
            return buf

        # Fast path: chunk is already in a slot (possibly free, possibly in-use).
        slot = self._tag_to_slot.get(chunk_id)
        if slot is not None:
            # Discard from the free set AND remove the stale node from
            # the deque so the two stay consistent. ``deque.remove`` is
            # O(N) but the buffer pool is small (n_buffer typically
            # ≤ 32), so eager cleanup is cheaper in practice than
            # letting stale nodes accumulate and re-pop them via the
            # ``popleft`` filter loop on the next miss.
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

        Oversize chunks (entries in :attr:`_large_buffers`) are dropped
        from the side-table here — there's no slot to return and no tag
        to preserve. This forfeits the forward→backward H2D-skip
        optimization for oversize chunks (the next gather will re-allocate
        and re-copy), but the bytes saved by freeing eagerly outweigh the
        rare reuse hit.

        After :meth:`close`, ``release`` is a silent no-op — close()
        has already dropped all pool state so there is nothing to
        return, and surfacing an error here would mask the real
        lifecycle bug at the matching acquire() call site (which is
        guarded explicitly).
        """
        if self._closed:
            return
        # Oversize fast path: free the side-table buffer eagerly when
        # the lease drops to zero. Done BEFORE the slot-pool path
        # because the two state spaces are disjoint by construction
        # (an oversize chunk never made it into ``_tag_to_slot``).
        # Mirrors the slot-pool decrement-then-drop discipline (lines
        # 348-358) so two simultaneous acquirers must each release
        # before the side-table entry is forgotten.
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

        Lease-free *peek* — the returned buffer pointer is only safe to
        read so long as no other ``acquire`` intervenes; if the slot
        happens to be in the free list (lease==0), the very next
        ``acquire()`` for a different chunk could evict the tag and
        overwrite the bytes.

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
        # Same lease-bookkeeping as the cache-hit fast path in ``acquire``.
        # Discard from the free set AND remove the stale node from the
        # deque so the two structures stay consistent (see ``acquire``
        # for the rationale).
        self._free_set.discard(slot)
        try:
            self._free.remove(slot)
        except ValueError:
            pass
        self._leases[slot] += 1
        return self._buffers[slot]

    def invalidate_tag(self, chunk_id: ChunkId) -> None:
        """Drop the GPU residency tag for ``chunk_id`` without recycling its slot.

        Force-evicts the cached chunk_id → slot mapping so the next
        ``acquire(chunk_id)`` is treated as a fresh miss (allocates a
        free slot and re-copies from CPU) instead of returning the
        currently-tagged buffer. The slot itself is left on whatever
        lease/free state it was in — only the *tag* is cleared.

        Used by :meth:`ChunkManager.restore_cpu_state` to ensure that a
        post-restore ``gather()`` for any restored chunk doesn't return
        stale GPU bytes from a buffer that was tagged before the CPU
        shadow was overwritten. Oversize chunks (entries in
        :attr:`_large_buffers`) are unaffected — their buffer carries
        the actual GPU bytes that would have to be re-copied; callers
        that need to invalidate those should drop their lease via
        :meth:`release` and let the lease-tracking path replace them on
        the next acquire.

        Safe to call when ``chunk_id`` has no current slot tag: the
        method silently returns. Raises ``RuntimeError`` if the slot is
        currently leased — invalidating in that state would leak the
        lease (the future ``release(chunk_id)`` would no-op because the
        chunk_id → slot mapping is gone, leaving the slot permanently
        in-use). The phase-2 restore call site holds no leases on the
        chunks it restores (the snapshot/restore window is outside any
        gather/release pair), so this guard surfaces the misuse case
        without blocking the legitimate caller.
        """
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
        # ``_free`` can carry stale entries (the lazy-cleanup path in
        # ``acquire``); ``_free_set`` is the authoritative count.
        return len(self._free_set)

    @property
    def num_in_use(self) -> int:
        return self.n_buffer - self.num_free

    def __len__(self) -> int:
        return self.n_buffer

    # ---- teardown ------------------------------------------------------

    def close(self) -> None:
        """Drop every pool-owned buffer and free the paired pinned region.

        Idempotent on success. Closes the paired :class:`PinnedHostMemory`
        region FIRST so a release failure (CUDA error, double-close from
        a misordered teardown, etc.) leaves the pool retryable; once
        host memory is actually released, the GPU-side bookkeeping is
        cleared and the pool is marked closed.

        If ``pinned_host.close()`` raises, the exception propagates and
        the pool state is left untouched — ``_closed`` stays ``False``,
        the slot tables remain intact, and ``self.pinned_host`` keeps
        its reference so the caller can retry teardown. Swallowing the
        error here would drop the only handle to the pinned allocation
        and leak it across a re-wrap (the runtime cannot reconstruct
        the handle once it goes out of scope).

        The pool is the sole owner of its ``pinned_host`` (constructed
        alongside the pool in :func:`_construct_runtime`) so closing it
        here is safe — no other component holds a borrow on the host
        slots after the chunk manager has released its own references.
        """
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
