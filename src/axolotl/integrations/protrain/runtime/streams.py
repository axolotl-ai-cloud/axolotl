"""Single-stream memory allocation context (paper Appendix B.2).

Status: **WIRED.** Production call sites now exist in
:mod:`chunk.buffer_pool` (long-lived pool slot pre-allocation),
:mod:`chunk.manager` (gather scratch / persistent buffers / restore-time
allocations), and the scheduler indirectly through those.

Paper reference
---------------
Appendix B.2 ("Single-Stream Memory Allocation") describes ProTrain
unifying every chunk-manager allocation onto the *default* CUDA stream
so the PyTorch caching allocator sees a single per-stream free list.
PyTorch's allocator maintains a separate heap per stream — a tensor
freed on stream A cannot be reused for an allocation on stream B
without ``record_stream`` hand-holding — and routing all allocations
through one stream sidesteps that fragmentation entirely.

What this module provides
-------------------------
:class:`SingleStreamAllocator` is a context manager that pins
Python-side allocations to a single managed CUDA stream (the default
stream by default). The ``__enter__`` / ``__exit__`` / ``sync()``
semantics compose with PyTorch's ``torch.cuda.stream(...)`` context
manager: a ``with SingleStreamAllocator():`` block nested inside an
outer ``with torch.cuda.stream(prefetch_stream):`` block correctly
switches the current stream to the default for the duration of the
inner block, then restores ``prefetch_stream`` on exit. Allocations
inside the inner block therefore land on the *default-stream* heap
even though kernel launches outside the inner block continue on the
prefetch stream.

Usage pattern (the App B.2 contract)
------------------------------------
The wire-up has two parts at every call site that hands a buffer to a
non-default stream:

1. **Allocate inside the allocator context** so the buffer comes from
   the default-stream heap::

       with torch.cuda.stream(prefetch_stream):
           with SingleStreamAllocator():
               buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)
           # ^ buf is on the default-stream heap
           buf.record_stream(prefetch_stream)
           buf.copy_(cpu_src, non_blocking=True)
           # ... use buf on prefetch_stream

2. **Call ``buf.record_stream(non_default_stream)`` immediately after
   exiting the allocator context** if the buffer is consumed on a
   non-default stream. This is the "directly managing deallocation
   synchronization ourselves" the paper mentions: it tells PyTorch's
   caching allocator to defer reuse of the buffer's underlying storage
   until ``non_default_stream`` has retired the work that uses it.
   Skipping this step opens a race where the allocator can hand the
   storage to a later default-stream allocation while the non-default
   stream is still reading or writing the bytes — silent data
   corruption.

Long-lived allocations (buffer-pool slots, persistent chunk buffers)
do not need ``record_stream`` because their lifetime is owned by the
manager and they are released only at teardown, when every consuming
stream has long since drained.

Reentrancy: the wrapper composes correctly with itself and with raw
``torch.cuda.stream(...)`` blocks. Like all CUDA-stream context
managers it is not thread-safe.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


class SingleStreamAllocator:
    """Context manager forcing allocations onto one managed CUDA stream.

    Usage::

        alloc = SingleStreamAllocator()  # uses the default stream
        with alloc:
            buf = torch.empty(...)
        alloc.sync()

    The context is a thin wrapper over ``torch.cuda.stream(stream)``:
    inside the ``with`` block the current stream is set to ``self.stream``
    so any allocations made from Python-side code land on that stream.
    Exiting the context restores the previous current stream — including
    when nested inside an outer ``torch.cuda.stream(prefetch_stream)``
    block, where the outer prefetch stream is restored on exit.

    See the module docstring for the App B.2 ``record_stream`` contract
    every call site that hands a freshly-allocated buffer to a
    non-default stream MUST observe.

    Reentrancy: the wrapper is safe to nest with itself and with raw
    ``torch.cuda.stream(...)`` blocks, but like all ``torch.cuda.stream``
    usage it is not thread-safe.
    """

    def __init__(self, stream: "torch.cuda.Stream | None" = None) -> None:
        # Import lazily so the module remains importable without a CUDA
        # runtime (matters for docs builds and syntax-only CI lanes).
        import torch

        self._torch = torch
        if stream is None:
            if not torch.cuda.is_available():
                LOG.debug(
                    "SingleStreamAllocator constructed without CUDA available; "
                    "stream operations will be no-ops."
                )
                self.stream: "torch.cuda.Stream | None" = None
            else:
                self.stream = torch.cuda.default_stream()
        else:
            self.stream = stream

        self._ctx_stack: list[AbstractContextManager[object]] = []

    def __enter__(self) -> "SingleStreamAllocator":
        if self.stream is None:
            return self
        ctx = self._torch.cuda.stream(self.stream)
        ctx.__enter__()
        self._ctx_stack.append(ctx)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._ctx_stack:
            return
        ctx = self._ctx_stack.pop()
        ctx.__exit__(exc_type, exc, tb)

    def sync(self) -> None:
        """Synchronize the managed stream.

        Blocks until every operation previously enqueued on ``self.stream``
        has completed. No-op if CUDA isn't available or no stream is set.
        """
        if self.stream is None:
            return
        self.stream.synchronize()


__all__ = ["SingleStreamAllocator"]
