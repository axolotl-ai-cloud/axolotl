"""Single-stream memory allocation context (paper Appendix B.2).

Status: **PARTIALLY IMPLEMENTED — class shipped, runtime wire-up deferred.**

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
:class:`SingleStreamAllocator` is a fully-functional context manager
that pins Python-side allocations to a single managed CUDA stream
(default stream by default). The ``__enter__`` / ``__exit__`` /
``sync()`` semantics are real and usable by any caller who explicitly
opts in.

Why this is a stub in practice
------------------------------
**The class is NOT currently wired into the ProTrain runtime.** A
search of the codebase shows zero call sites outside this module's own
docstring example. In particular, :mod:`runtime.scheduler` does the
*opposite* of the paper's design: it constructs two dedicated
non-default ``torch.cuda.Stream()`` instances — one for chunk prefetch,
one for activation swap — so that PCIe copies can overlap the compute
stream. See ``runtime/scheduler.py::_init_streams`` (the
``self._prefetch_stream = torch.cuda.Stream()`` /
``self._swap_stream = torch.cuda.Stream()`` lines).

This is a known, intentional deviation from the paper, tracked in
``DESIGN.md`` under
"Design Decisions → Single-Stream Allocation (App B.2 — DEFERRED)".
The wire-up is non-trivial: every allocation in the chunk manager,
buffer pool, and scheduler would need to route through one allocator
context, and buffer-pool reuse semantics would need to be reverified.
Profile data on the project's reference hardware (RTX 3090) has not
shown allocator fragmentation as a top contributor to peak memory —
the ``α = 1.10`` fragmentation factor (see DESIGN.md §Design Decisions
item 1) covers it — so the work is deferred until a measurement
justifies it.

Calling this class
------------------
A caller who imports and instantiates :class:`SingleStreamAllocator`
gets a real allocator context, but they will trigger a
:class:`UserWarning` at construction time pointing back here, because
no production caller exists today and accidental future use should be
discoverable in logs.
"""

from __future__ import annotations

import warnings
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)

# Module-level flag so the deferred-wire-up warning fires only once per
# process, no matter how many SingleStreamAllocator instances get
# constructed. Keeps test logs and any future accidental call sites
# readable.
_WARNED_UNWIRED = False


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
    Exiting the context restores the previous current stream.

    Reentrancy: the wrapper is safe to nest with itself, but like all
    ``torch.cuda.stream`` usage it is not thread-safe.

    .. warning::
        This class corresponds to paper Appendix B.2 but is **not wired
        into the ProTrain runtime** — the scheduler uses dedicated
        non-default streams. Constructing an instance emits a one-time
        :class:`UserWarning` so accidental future callers can find this
        gap via logs. See the module docstring and ``DESIGN.md``
        §"Single-Stream Allocation (App B.2 — DEFERRED)" for the full
        rationale.
    """

    def __init__(self, stream: "torch.cuda.Stream | None" = None) -> None:
        global _WARNED_UNWIRED
        if not _WARNED_UNWIRED:
            _WARNED_UNWIRED = True
            warnings.warn(
                "SingleStreamAllocator is defined for paper App B.2 fidelity but is "
                "not wired into the ProTrain runtime. See DESIGN.md "
                "§Single-Stream Allocation for status.",
                stacklevel=2,
            )

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
