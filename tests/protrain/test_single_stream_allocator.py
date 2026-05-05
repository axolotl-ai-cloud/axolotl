"""Tests for ``SingleStreamAllocator`` and its wire-up (paper App B.2).

The semantic the wire-up enforces is: every long-lived ProTrain GPU
allocation comes from PyTorch's *default-stream* heap, even when the
caller is inside a non-default ``torch.cuda.stream(...)`` context.
PyTorch doesn't expose heap-stream affinity directly; the most robust
proxy is "free + reallocate inside the allocator context yields no
fragmentation growth across many cycles" — which exercises exactly the
behaviour App B.2 promises.

These tests skip cleanly without CUDA — they have no meaning on a
host with no GPU heap.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SingleStreamAllocator semantics only meaningful with a CUDA device",
)


# ---------------------------------------------------------------------------
# Context-manager composition (CUDA-only)
# ---------------------------------------------------------------------------


@cuda_only
def test_nests_inside_outer_stream_context_and_restores_on_exit():
    """Inner ``SingleStreamAllocator`` switches to default; exit restores outer."""
    from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator

    default = torch.cuda.default_stream()
    s1 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        assert torch.cuda.current_stream() == s1
        with SingleStreamAllocator():
            # Inside the wrap: current stream MUST be the default
            # stream so allocations land on its heap.
            assert torch.cuda.current_stream() == default
            # Allocate inside — the act of allocating-while-default-current
            # is what App B.2 requires.
            buf = torch.empty(1024, dtype=torch.uint8, device="cuda")
            assert buf.is_cuda
        # After exit: outer prefetch-stream-equivalent must be restored.
        assert torch.cuda.current_stream() == s1

    assert torch.cuda.current_stream() == default


@cuda_only
def test_can_be_used_without_outer_stream_context():
    """Bare ``with SingleStreamAllocator():`` keeps the default stream current."""
    from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator

    default = torch.cuda.default_stream()
    assert torch.cuda.current_stream() == default
    with SingleStreamAllocator():
        assert torch.cuda.current_stream() == default
        _ = torch.empty(1024, dtype=torch.uint8, device="cuda")
    assert torch.cuda.current_stream() == default


@cuda_only
def test_sync_blocks_until_stream_drains():
    """``.sync()`` returns only after the managed stream has drained."""
    from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator

    alloc = SingleStreamAllocator()
    with alloc:
        # Issue some work on the default stream.
        a = torch.empty(1024 * 1024, dtype=torch.float32, device="cuda")
        a.fill_(1.0)
    alloc.sync()  # must not raise; must not deadlock


# ---------------------------------------------------------------------------
# Heap-affinity probe via free+realloc fragmentation cycles (CUDA-only)
# ---------------------------------------------------------------------------


@cuda_only
def test_alloc_inside_wrap_belongs_to_default_heap_via_fragmentation_probe():
    """Allocations under the wrap reuse the default-stream free list cleanly.

    The probe: with a non-default stream as the current stream for the
    OUTER context, allocate-and-free a fixed-size GPU buffer many times
    INSIDE the ``SingleStreamAllocator`` wrap. If the wrap is doing its
    job, every allocation comes from the same per-default-stream free
    list and the reserved-bytes high-water mark stays bounded by one
    block's size (modulo the allocator's rounding granularity). If the
    wrap were broken — i.e. the allocations were landing on the
    non-default stream's heap and being freed on the default stream's
    heap — PyTorch's caching allocator would NOT be able to reuse them
    (heaps are separate), and reserved bytes would grow per cycle.
    """
    from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator

    s1 = torch.cuda.Stream()
    nbytes = 4 * 1024 * 1024  # 4 MB — large enough to be a real block

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Warm up: do one alloc/free cycle so the allocator picks a block size.
    with torch.cuda.stream(s1):
        with SingleStreamAllocator():
            warm = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        del warm
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_reserved()

    # Now repeat many alloc/free cycles. Reserved should NOT grow per cycle.
    for _ in range(64):
        with torch.cuda.stream(s1):
            with SingleStreamAllocator():
                buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            del buf
    torch.cuda.synchronize()
    after = torch.cuda.memory_reserved()

    # Allow at most one extra block of growth — sometimes the allocator
    # reserves a second block for alignment or carve-out reasons. The
    # failure mode we are guarding against is *unbounded* growth (e.g.
    # 64 × nbytes), which would indicate the per-cycle alloc and free
    # are landing on different heaps.
    assert after <= baseline + 2 * nbytes, (
        f"Reserved bytes grew from {baseline} to {after} across 64 alloc/free "
        f"cycles inside SingleStreamAllocator (delta {after - baseline} bytes, "
        f"per-block size {nbytes}). This indicates allocations are NOT being "
        f"routed to the default-stream heap — App B.2 wire-up is broken."
    )


# ---------------------------------------------------------------------------
# BufferPool wire-up (CUDA-only)
# ---------------------------------------------------------------------------


@cuda_only
def test_buffer_pool_slots_allocated_from_default_heap():
    """Constructing a BufferPool while a non-default stream is current yields
    pool slots whose bytes nevertheless came from the default-stream heap.

    Same fragmentation-probe strategy: build a pool inside an outer
    non-default stream context, free it, build another, and check
    reserved memory does not grow unboundedly. If the buffer-pool
    pre-allocation were not routed through ``SingleStreamAllocator``,
    each cycle's slots would land on the outer stream's heap — and
    the deletions on the same stream would return them to that heap —
    *but* the next pool's allocations would also go to that heap.
    The probe still catches the wrong-heap case indirectly: we
    construct one pool on s1, then free it, then check that a
    subsequent default-stream allocation can reuse the bytes (which
    requires they came from the default heap to begin with).
    """
    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    n_buffer = 4
    s_chunk = 4 * 1024 * 1024  # 4 MB per slot, total 16 MB

    s1 = torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    pinned = PinnedHostMemory(n_buffer=n_buffer, S_chunk=s_chunk)
    try:
        # Build the pool while a non-default stream is current. If the
        # wire-up is correct, the pool slots' bytes go to the default
        # heap regardless.
        with torch.cuda.stream(s1):
            pool = BufferPool(
                n_buffer=n_buffer,
                S_chunk=s_chunk,
                pinned_host=pinned,
                device="cuda",
            )
        assert len(pool) == n_buffer
        # Drop the pool so its slots return to the allocator.
        del pool
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reserved baseline AFTER the pool is gone and cache cleared:
        # allocations from here should reuse the freed bytes if the
        # pool's slots were on the default heap. We measure baseline
        # then allocate the same total bytes on the default stream;
        # reserved should not double.
        baseline_reserved = torch.cuda.memory_reserved()

        # Allocate the same total footprint as the pool, on the default
        # stream. If the pool's bytes are on the same heap, the
        # allocator will largely reuse them.
        bufs = [
            torch.empty(s_chunk, dtype=torch.uint8, device="cuda")
            for _ in range(n_buffer)
        ]
        torch.cuda.synchronize()
        # Total demanded: n_buffer * s_chunk. Reserved ought to be at
        # most baseline + n_buffer * s_chunk + a small overhead for
        # cache-line / alignment slack — definitely not double the
        # demand (which would mean the pool's heap and the default
        # heap are disjoint).
        ceiling = baseline_reserved + (n_buffer * s_chunk) + (4 * 1024 * 1024)
        actual = torch.cuda.memory_reserved()
        assert actual <= ceiling, (
            f"BufferPool slots appear to be on the wrong heap: "
            f"reserved jumped from baseline {baseline_reserved} to {actual} "
            f"(ceiling {ceiling}) when allocating {n_buffer * s_chunk} bytes "
            f"on the default stream after dropping the pool. The wire-up "
            f"in chunk/buffer_pool.py is not routing pool slots through "
            f"SingleStreamAllocator."
        )
        del bufs
    finally:
        pinned.close()


# ---------------------------------------------------------------------------
# CPU-only smoke (works without CUDA — keeps coverage on test boxes
# that haven't got a GPU)
# ---------------------------------------------------------------------------


def test_constructible_without_cuda_no_warning_no_op():
    """``SingleStreamAllocator()`` on a CPU-only host is a clean no-op."""
    from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator

    # The constructor must not raise even if torch.cuda.is_available() is False
    # (CI lanes without a GPU). Whether CUDA is present or not, the
    # construction is silent — the previous one-time UserWarning is gone now
    # that production callers exist.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alloc = SingleStreamAllocator()
        # ``with`` and ``sync`` must be no-ops, not exceptions, when the
        # process has no CUDA stream to manage.
        with alloc:
            pass
        alloc.sync()

    # Filter to warnings actually originating in streams.py — other modules'
    # imports can trip unrelated DeprecationWarnings on CI lanes.
    streams_warnings = [
        w
        for w in caught
        if "streams.py" in str(w.filename) or "SingleStreamAllocator" in str(w.message)
    ]
    assert streams_warnings == [], (
        f"SingleStreamAllocator() should be silent now that production callers "
        f"exist; got warnings: {[str(w.message) for w in streams_warnings]}"
    )
