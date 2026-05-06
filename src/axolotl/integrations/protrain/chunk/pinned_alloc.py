"""Precise-size pinned host memory (Appendix B.2).

PyTorch's default ``CUDAHostAllocator`` rounds up pinned allocations to the
next power of two. For ``n_buffer * S_chunk`` that can waste hundreds of MB
on large chunks. We instead call ``cudaHostAlloc`` directly through
``ctypes`` for an exact byte count, and hand out zero-copy ``torch.Tensor``
views over the resulting buffer.

If the ``libcudart`` lookup fails (e.g. the system's CUDA runtime isn't
visible to ``ctypes.CDLL`` despite ``torch.cuda`` being available), we fall
back to ``torch.empty(size, pin_memory=True)`` and flag
``_is_precise_size = False`` so tests can detect and skip assertions that
depend on exact sizing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)

# cudaHostAllocDefault from cuda_runtime_api.h: "Default page-locked allocation flag".
_CUDA_HOST_ALLOC_DEFAULT = 0
_CUDA_SUCCESS = 0


def _load_cudart() -> ctypes.CDLL | None:
    """Locate ``libcudart`` as a ``ctypes.CDLL`` handle; return None if unavailable.

    On recent PyTorch builds ``torch.cuda.cudart()`` returns a Python module
    (``torch._C._cudart``) rather than a ``ctypes.CDLL`` — the symbols are
    not the raw C functions we need to set ``argtypes``/``restype`` on, so
    we skip that path entirely and load the shared object directly via
    ``ctypes``. We try a handful of common SONAMEs (CUDA 11, 12, 13) and
    finally ``ctypes.util.find_library('cudart')`` which resolves to
    whichever ``libcudart.so.*`` ``ldconfig`` knows about.
    """
    # Prefer the runtime that matches the PyTorch build when discoverable —
    # mixing torch's compiled-against major with a different ``libcudart`` on
    # the search path is a known compatibility hazard, so we try the matching
    # SONAME first before falling back to the deterministic newest-first list.
    try:
        import torch

        cuda_version = torch.version.cuda
    except Exception:  # noqa: BLE001
        cuda_version = None

    # Explicit versioned SONAMEs follow so we prefer a specific major
    # version (and a deterministic newest-first order) when more than one
    # runtime is on the library search path. ``libcudart.so`` is the
    # unversioned symlink (only present with -dev packages) and is tried
    # last as a fallback for systems where the versioned SONAME isn't
    # directly resolvable but the dev symlink is.
    candidates: list[str] = []
    if cuda_version:
        major = cuda_version.split(".", maxsplit=1)[0]
        candidates.append(f"libcudart.so.{major}")
    candidates.extend(
        [
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "libcudart.so",
        ]
    )
    # Let ctypes locate whatever the current ld cache has, too.
    resolved = ctypes.util.find_library("cudart")
    if resolved:
        candidates.append(resolved)

    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class PinnedHostMemory:
    """One large precise-size pinned host allocation split into ``n_buffer`` slots.

    Memory is allocated once in ``__init__`` and freed once in ``__del__``
    (or via :meth:`close`). Slots are contiguous and identically sized —
    ``buffer(i)`` hands out the ``i``-th slot as a pinned ``torch.Tensor``.

    Lifetime hazard
    ---------------
    ``buffer(i)`` returns a ``narrow()`` view sharing storage with the
    underlying pinned region. If ``close()`` is called while a caller
    still holds such a view, the view becomes a dangling pointer —
    subsequent reads/writes (including async H2D copies) will touch
    freed memory. To guard against this, ``buffer(i)`` increments a
    borrow counter that the caller must decrement via
    :meth:`release_buffer` once the slot is no longer in use (the
    canonical pattern is acquire-via-``buffer`` then
    ``record_stream`` + ``release_buffer`` after enqueueing the H2D
    copy). :meth:`close` raises ``RuntimeError`` if any borrow is
    still outstanding so the lifetime violation is loud rather than
    silent. Destructor-driven cleanup (:meth:`__del__`) cannot raise,
    so it instead **intentionally leaks** the pinned region until
    process exit when borrows are still outstanding (logging loudly
    so the missing ``release_buffer`` is diagnosable); it only frees
    when no borrows remain.
    """

    def __init__(self, n_buffer: int, S_chunk: int) -> None:
        if n_buffer <= 0:
            raise ValueError(f"n_buffer must be positive, got {n_buffer}")
        if S_chunk <= 0:
            raise ValueError(f"S_chunk must be positive, got {S_chunk}")

        self.n_buffer = int(n_buffer)
        self.S_chunk = int(S_chunk)
        self.total_bytes = self.n_buffer * self.S_chunk

        # Reentrant lock guarding the shared lifecycle state below
        # (``_closed``, ``_torch_tensor``, ``_fallback_tensor``,
        # ``_live_borrows``). ``buffer`` / ``release_buffer`` / ``close``
        # may be called from different threads under the swap pipeline
        # (e.g. a worker doing H2D copies vs. the trainer thread tearing
        # the allocator down) and the check-then-use sequences (``if not
        # self._closed`` followed by ``self._torch_tensor.narrow(...)``,
        # for example) must be atomic to avoid racing teardown.
        # ``RLock`` so a method holding the lock can call another locked
        # method on the same instance without deadlocking.
        self._lock = threading.RLock()

        self._cudart: ctypes.CDLL | None = None
        self._ptr: int = 0  # device-facing pointer value (host-side VA)
        self._closed = False
        self._fallback_tensor: "torch.Tensor | None" = None
        self._torch_tensor: "torch.Tensor | None" = None
        self._is_precise_size: bool = False
        # Per-slot borrow counts: ``{slot_idx: outstanding_borrows}``.
        # ``buffer(i)`` increments ``_live_borrows[i]``; ``release_buffer(i)``
        # decrements it (and prunes the key when it hits zero so ``close()``'s
        # check is "is this dict non-empty"). A per-slot map (rather than a
        # single global counter) lets the pool answer *which* slot is still
        # live, which the swap pipeline needs to gate event-based release of
        # individual slots without conflating concurrent borrows on others.
        # Reentrant / multi-borrow semantics on the same slot are supported
        # (count-per-slot, not set-of-live-slots) because callers may stage
        # overlapping H2D copies on the same slot during pipelined refill.
        self._live_borrows: dict[int, int] = {}

        cudart = _load_cudart()
        if cudart is None:
            LOG.warning(
                "PinnedHostMemory: libcudart not found via ctypes; "
                "falling back to torch.empty(pin_memory=True). "
                "Pinned buffer may be rounded to a power of two."
            )
            self._init_fallback()
            return

        try:
            self._init_cudart(cudart)
        except Exception as err:  # noqa: BLE001
            # If ``cudaHostAlloc`` succeeded but a follow-up step
            # (``torch.frombuffer``, attribute setup, etc.) raised, ``_ptr``
            # is populated and the pinned region is live. Without an explicit
            # free here, ``_init_fallback()`` would allocate a *second*
            # backing store of the same size — a transient double allocation
            # that can OOM construction on large chunks. Drop the partially
            # initialized buffer first so the fallback path starts clean.
            if self._cudart is not None and self._ptr:
                free_status = self._cudart.cudaFreeHost(ctypes.c_void_p(self._ptr))
                if free_status != _CUDA_SUCCESS:
                    LOG.warning(
                        "cudaFreeHost during cudart-init cleanup returned status=%d",
                        free_status,
                    )
            self._cudart = None
            self._ptr = 0
            self._torch_tensor = None
            self._is_precise_size = False
            LOG.warning(
                "PinnedHostMemory: ctypes cudaHostAlloc path failed (%s); "
                "falling back to torch.empty(pin_memory=True).",
                err,
            )
            self._init_fallback()

    # ---- initialization paths ------------------------------------------

    def _init_cudart(self, cudart: ctypes.CDLL) -> None:
        import torch

        # cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
        try:
            cudart.cudaHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            cudart.cudaHostAlloc.restype = ctypes.c_int
            cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
            cudart.cudaFreeHost.restype = ctypes.c_int
        except AttributeError as err:
            raise RuntimeError(f"cudart missing required symbol: {err}") from err

        ptr = ctypes.c_void_p(0)
        status = cudart.cudaHostAlloc(
            ctypes.byref(ptr),
            ctypes.c_size_t(self.total_bytes),
            ctypes.c_uint(_CUDA_HOST_ALLOC_DEFAULT),
        )
        if status != _CUDA_SUCCESS or not ptr.value:
            raise RuntimeError(
                f"cudaHostAlloc returned status={status} ptr={ptr.value} "
                f"for size={self.total_bytes}"
            )

        self._cudart = cudart
        self._ptr = int(ptr.value)
        self._is_precise_size = True

        # Build a single torch.Tensor viewing the whole region as uint8. We
        # use ``torch.frombuffer`` on a ``ctypes`` array cast so the tensor
        # shares storage with our cudaHostAlloc'd region with no copy.
        ArrayT = ctypes.c_uint8 * self.total_bytes
        # ``ArrayT.from_address(ptr)`` produces a ctypes array backed by the
        # pinned host region. ``torch.frombuffer`` takes any object that
        # supports the buffer protocol and exposes it as a zero-copy tensor.
        buf = ArrayT.from_address(self._ptr)
        self._torch_tensor = torch.frombuffer(buf, dtype=torch.uint8)
        # The buffer-protocol path doesn't carry the ``pin_memory`` flag
        # because PyTorch only sets that for allocations it made itself.
        # The underlying memory IS pinned (we called cudaHostAlloc), just
        # torch can't prove it. ``is_pinned()`` will therefore return False
        # on this path despite the memory being physically pinned —
        # consequently ``copy_(..., non_blocking=True)`` H2D copies that
        # source from a slot view will silently fall back to BLOCKING
        # behaviour, defeating the throughput rationale for cudaHostAlloc.
        # We loudly warn at construction so callers don't get silent
        # serialization, and expose ``is_pinned_recognised_by_torch``
        # below for callers that want to gate non_blocking on the
        # honest answer.
        LOG.warning(
            "PinnedHostMemory: cudaHostAlloc'd region is physically pinned, "
            "but torch.frombuffer-built tensors report is_pinned()=False. "
            "copy_(..., non_blocking=True) H2D from these slots will fall "
            "back to blocking copies — see is_pinned_recognised_by_torch."
        )

    def _init_fallback(self) -> None:
        import torch

        # ``pin_memory=True`` requires a working CUDA driver; on CPU-only
        # hosts the call raises. Gate on availability so unit tests + CI
        # without a GPU can still exercise the fallback path with
        # paged host memory.
        pin = bool(torch.cuda.is_available())
        self._fallback_tensor = torch.empty(
            self.total_bytes, dtype=torch.uint8, pin_memory=pin
        )
        self._torch_tensor = self._fallback_tensor
        self._is_precise_size = False

    # ---- public API ----------------------------------------------------

    @property
    def is_precise_size(self) -> bool:
        """True iff the underlying bytes == exactly ``n_buffer * S_chunk``."""
        return self._is_precise_size

    @property
    def is_pinned_recognised_by_torch(self) -> bool:
        """True iff ``torch.is_pinned()`` on a slot view will return True.

        On the ctypes / ``cudaHostAlloc`` precise-size path this is
        ``False`` — the memory IS pinned, but ``torch.frombuffer`` does
        not propagate the ``pin_memory`` flag, so ``tensor.is_pinned()``
        reports ``False`` and PyTorch's ``copy_(..., non_blocking=True)``
        guard treats the source as pageable and silently falls back to
        a blocking copy.

        On the ``torch.empty(pin_memory=True)`` fallback path (and on the
        non-CUDA paged fallback) this matches whatever torch reports for
        the underlying tensor — typically ``True`` on a CUDA host, but
        we don't probe ``is_pinned()`` here because that would require
        torch import at attribute-access time. Callers that need the
        post-construction probe should call ``buffer(i).is_pinned()``
        directly.
        """
        # Only the ctypes path has the ``is_pinned()`` blind spot. The
        # fallback path was constructed via ``torch.empty(pin_memory=...)``
        # and torch knows about its own allocation.
        return not self._is_precise_size

    def buffer(self, i: int) -> "torch.Tensor":
        """Return the ``i``-th slot as a 1D ``uint8`` tensor of length ``S_chunk``.

        The returned view shares storage with the pinned region; writes are
        immediately visible to CUDA transfers that use the same host pointer.

        The slot is considered borrowed until the caller pairs this call
        with :meth:`release_buffer`. ``close()`` will refuse to free the
        underlying pinned region while any borrow is still outstanding
        (see the class docstring for the use-after-free hazard).

        Pinned-recognition limitation
        -----------------------------
        On the precise-size ctypes / ``cudaHostAlloc`` path the returned
        tensor's ``is_pinned()`` reports ``False`` even though the
        underlying memory is physically pinned (``torch.frombuffer``
        does not propagate ``pin_memory``). This means
        ``dst_gpu.copy_(slot_view, non_blocking=True)`` will silently
        degrade to a blocking copy — torch's ``non_blocking`` fast path
        is gated on ``is_pinned()``, not on whether the OS marked the
        page as pinned. Callers that depend on overlap of H2D with
        compute MUST consult :attr:`is_pinned_recognised_by_torch` and
        either accept the blocking fallback or use the
        ``torch.empty(pin_memory=True)`` fallback path (precise-size off)
        where torch's bookkeeping is intact.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("PinnedHostMemory is closed")
            if not 0 <= i < self.n_buffer:
                raise IndexError(f"buffer index {i} out of range [0, {self.n_buffer})")
            assert self._torch_tensor is not None
            start = i * self.S_chunk
            view = self._torch_tensor.narrow(0, start, self.S_chunk)
            self._live_borrows[i] = self._live_borrows.get(i, 0) + 1
            return view

    def release_buffer(self, i: int) -> None:
        """Decrement the borrow count for slot ``i``.

        Pairs with :meth:`buffer`. The per-slot count is the ownership
        signal :meth:`close` consults; failing to release leaves
        ``close()`` raising. Index validation is best-effort so this
        is safe to call from cleanup paths even if the slot id was
        never borrowed in this allocator (logged but not fatal — we
        prefer not to derail destructor flows).
        """
        if not 0 <= i < self.n_buffer:
            LOG.warning(
                "PinnedHostMemory.release_buffer: index %d out of range "
                "[0, %d); ignored",
                i,
                self.n_buffer,
            )
            return
        with self._lock:
            count = self._live_borrows.get(i, 0)
            if count <= 0:
                LOG.warning(
                    "PinnedHostMemory.release_buffer(%d): no outstanding borrow "
                    "for that slot; double-release?",
                    i,
                )
                return
            if count == 1:
                # Prune so ``_live_borrows`` is empty iff every slot is
                # released — makes ``close()``'s check a simple truthiness
                # test on the dict.
                del self._live_borrows[i]
            else:
                self._live_borrows[i] = count - 1

    # ---- introspection helpers (additive; backwards compatible) -----------

    def borrow_count(self, i: int) -> int:
        """Return the number of outstanding borrows on slot ``i`` (0 if none).

        Additive helper for callers (e.g. the swap pipeline) that need to
        reason about per-slot lifetime — the previous global counter could
        not distinguish which slot was live.
        """
        if not 0 <= i < self.n_buffer:
            return 0
        with self._lock:
            return self._live_borrows.get(i, 0)

    def live_slots(self) -> list[int]:
        """Return slot indices with at least one outstanding borrow.

        Order is unspecified. Useful for diagnostics and for the swap
        pipeline's event-based release flow, which needs to enumerate which
        slots are still in flight.
        """
        with self._lock:
            return list(self._live_borrows.keys())

    @property
    def total_live_borrows(self) -> int:
        """Aggregate borrow count across all slots.

        Preserves the semantics of the prior ``_live_borrows`` integer for
        any external caller that only cared about "is anything still
        borrowed" — though :meth:`live_slots` is preferred for new code.
        """
        with self._lock:
            return sum(self._live_borrows.values())

    def close(self) -> None:
        """Free the pinned allocation. Idempotent.

        Raises ``RuntimeError`` if any slot view returned by
        :meth:`buffer` has not been returned via :meth:`release_buffer`
        — freeing the underlying pinned region while views are still
        live can create dangling pointers and silently corrupt any
        in-flight H2D copy or host write that targets the slot. The
        explicit ``close()`` path is the user-controlled deterministic
        teardown surface, so we want loud failure on lifetime
        violations. Destructor-driven cleanup falls through
        :meth:`__del__`, which logs and intentionally skips free when
        borrows remain (to avoid use-after-free), and only frees when
        no borrows are outstanding.
        """
        with self._lock:
            if self._closed:
                return
            if self._live_borrows:
                outstanding = sum(self._live_borrows.values())
                slots = sorted(self._live_borrows.keys())
                raise RuntimeError(
                    f"PinnedHostMemory.close(): {outstanding} slot view(s) "
                    f"still borrowed across slots {slots}; release them via "
                    "release_buffer() before close() to avoid use-after-free "
                    "on the pinned region."
                )
            self._closed = True
            # Drop torch views first so no tensor outlives the underlying
            # memory.
            self._torch_tensor = None
            self._fallback_tensor = None
            if self._cudart is not None and self._ptr:
                status = self._cudart.cudaFreeHost(ctypes.c_void_p(self._ptr))
                if status != _CUDA_SUCCESS:
                    LOG.warning("cudaFreeHost returned status=%d", status)
                self._ptr = 0
                self._cudart = None

    def __del__(self) -> None:  # noqa: D401
        # Destructors must not throw, so the borrow guard in ``close()``
        # is bypassed here. But if borrows are still outstanding when the
        # allocator is garbage-collected, the user has an ownership bug:
        # views (or async H2D copies) referencing the pinned region are
        # still live. Force-freeing here would convert that ownership bug
        # into a use-after-free / dangling-pointer scenario where the next
        # touch of the slot reads or writes already-released memory and
        # may silently corrupt unrelated allocations. The safer choice in
        # the destructor path is to *leak* the pinned region until process
        # teardown reclaims it: the OS will free it, and the leak is loudly
        # logged so the missing ``release_buffer`` is diagnosable. Only
        # when no borrows remain do we proceed to the deterministic
        # ``close()`` free.
        try:
            # ``_lock`` may not exist if __init__ raised before its
            # creation. ``getattr`` with a no-op fallback keeps the
            # destructor robust to partial construction failures.
            lock = getattr(self, "_lock", None)
            if lock is not None:
                lock.acquire()
            try:
                if self._closed:
                    return
                if self._live_borrows:
                    LOG.warning(
                        "PinnedHostMemory.__del__: %d slot view(s) still "
                        "borrowed across slots %s at GC time; leaking pinned "
                        "region until process exit to avoid dangling-pointer "
                        "use-after-free. Caller is missing release_buffer() "
                        "pairs — fix the ownership bug and call close() "
                        "explicitly.",
                        sum(self._live_borrows.values()),
                        sorted(self._live_borrows.keys()),
                    )
                    return
            finally:
                if lock is not None:
                    lock.release()
            # ``close()`` re-acquires the lock; release before calling so
            # the RLock count returns to zero cleanly.
            self.close()
        except Exception:  # noqa: BLE001 — destructors must not throw
            LOG.exception("Error during PinnedHostMemory.__del__ cleanup")


__all__ = ["PinnedHostMemory"]
