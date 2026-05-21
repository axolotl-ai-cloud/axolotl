"""Precise-size pinned host memory via cudaHostAlloc + ctypes; falls back to torch pin_memory."""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)

# cudaHostAllocDefault flag.
_CUDA_HOST_ALLOC_DEFAULT = 0
_CUDA_SUCCESS = 0


def _load_cudart() -> ctypes.CDLL | None:
    """Locate ``libcudart`` via ctypes (prefer torch's compiled-against CUDA major)."""
    try:
        import torch

        cuda_version = torch.version.cuda
    except Exception:  # noqa: BLE001
        cuda_version = None

    # Versioned SONAMEs in newest-first order; unversioned symlink last (dev-only).
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
    """One precise-size pinned host allocation split into n_buffer slots.

    Lifetime hazard: buffer(i) returns a narrow() view; close() while views are live
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

        # RLock for atomic check-then-use over closed/tensor/borrows under concurrent close.
        self._lock = threading.RLock()

        self._cudart: ctypes.CDLL | None = None
        self._ptr: int = 0  # device-facing pointer value (host-side VA)
        self._closed = False
        self._fallback_tensor: "torch.Tensor | None" = None
        self._torch_tensor: "torch.Tensor | None" = None
        # Anchors the ctypes buffer-protocol view alive across cudaFreeHost.
        self._cudart_view: "torch.Tensor | None" = None
        self._is_precise_size: bool = False
        # torch.is_pinned() result; False → non_blocking H2D silently blocks.
        self._is_pinned_recognised_by_torch: bool = False
        # Per-slot borrow counts; close() refuses while dict is non-empty.
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
            # Free partial cudaHostAlloc region before fallback to avoid double-alloc OOM.
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
            self._cudart_view = None
            self._is_precise_size = False
            self._is_pinned_recognised_by_torch = False
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

        # Zero-copy uint8 tensor view via torch.frombuffer over ctypes array.
        ArrayT = ctypes.c_uint8 * self.total_bytes
        buf = ArrayT.from_address(self._ptr)
        frombuffer_tensor = torch.frombuffer(buf, dtype=torch.uint8)

        # Probe is_pinned(); buffer-protocol historically doesn't propagate the flag.
        try:
            recognised = bool(frombuffer_tensor.is_pinned())
        except Exception:  # noqa: BLE001 — be resilient to torch quirks
            recognised = False

        if recognised:
            # Fast path: zero-copy view; non_blocking H2D overlaps.
            self._torch_tensor = frombuffer_tensor
            self._is_pinned_recognised_by_torch = True
        else:
            # Hand off to torch.empty(pin_memory=True) so non_blocking H2D works; free cudaHostAlloc.
            LOG.warning(
                "PinnedHostMemory: torch.frombuffer view of the "
                "cudaHostAlloc'd region reports is_pinned()=False on "
                "this PyTorch build. Allocating a parallel "
                "torch.empty(pin_memory=True) buffer so non_blocking "
                "H2D copies actually overlap, then freeing the original "
                "cudaHostAlloc region so host-side pinned footprint "
                "stays single-counted."
            )
            torch_pinned = torch.empty(
                self.total_bytes, dtype=torch.uint8, pin_memory=True
            )
            torch_pinned.copy_(frombuffer_tensor)
            # Drop ctypes-backed view before cudaFreeHost.
            del frombuffer_tensor
            del buf
            free_status = cudart.cudaFreeHost(ctypes.c_void_p(self._ptr))
            if free_status != _CUDA_SUCCESS:
                LOG.warning(
                    "cudaFreeHost (post-fallback handover) returned status=%d; "
                    "leaking the original cudaHostAlloc region",
                    free_status,
                )
            # Sentinel _ptr=0/_cudart=None tells close() not to double-free.
            self._ptr = 0
            self._cudart = None
            self._cudart_view = None
            self._torch_tensor = torch_pinned
            # torch.empty pin_memory: pinned-recognised, but not precise-size.
            self._is_pinned_recognised_by_torch = True
            self._is_precise_size = False

    def _init_fallback(self) -> None:
        import torch

        # pin_memory=True requires CUDA; CPU-only hosts use paged memory.
        pin = bool(torch.cuda.is_available())
        self._fallback_tensor = torch.empty(
            self.total_bytes, dtype=torch.uint8, pin_memory=pin
        )
        self._torch_tensor = self._fallback_tensor
        self._is_precise_size = False
        # torch-allocated → is_pinned() matches pin.
        self._is_pinned_recognised_by_torch = pin

    # ---- public API ----------------------------------------------------

    @property
    def is_precise_size(self) -> bool:
        """True iff the underlying bytes == exactly ``n_buffer * S_chunk``."""
        return self._is_precise_size

    @property
    def is_pinned_recognised_by_torch(self) -> bool:
        """True iff torch.is_pinned() on a slot view returns True."""
        return self._is_pinned_recognised_by_torch

    def buffer(self, i: int) -> "torch.Tensor":
        """Return the i-th slot as a 1D uint8 tensor of length S_chunk; pair with release_buffer."""
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
        """Decrement the borrow count for slot ``i`` (pairs with buffer())."""
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
                # Prune so close() can use a simple truthiness check on the dict.
                del self._live_borrows[i]
            else:
                self._live_borrows[i] = count - 1

    # ---- introspection helpers (additive; backwards compatible) -----------

    def borrow_count(self, i: int) -> int:
        """Return the number of outstanding borrows on slot ``i`` (0 if none)."""
        if not 0 <= i < self.n_buffer:
            return 0
        with self._lock:
            return self._live_borrows.get(i, 0)

    def live_slots(self) -> list[int]:
        """Return slot indices with at least one outstanding borrow (order unspecified)."""
        with self._lock:
            return list(self._live_borrows.keys())

    @property
    def total_live_borrows(self) -> int:
        """Aggregate borrow count across all slots."""
        with self._lock:
            return sum(self._live_borrows.values())

    def close(self) -> None:
        """Free the pinned allocation; raises if any borrow is outstanding. Idempotent."""
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
            # Drop torch views before cudaFreeHost.
            self._torch_tensor = None
            self._fallback_tensor = None
            self._cudart_view = None
            if self._cudart is not None and self._ptr:
                status = self._cudart.cudaFreeHost(ctypes.c_void_p(self._ptr))
                if status != _CUDA_SUCCESS:
                    LOG.warning("cudaFreeHost returned status=%d", status)
                self._ptr = 0
                self._cudart = None

    def __del__(self) -> None:  # noqa: D401
        # On outstanding borrows: leak the region rather than force-free into use-after-free.
        try:
            # _lock may not exist if __init__ raised early.
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
            # close() re-acquires the lock; release first so RLock count clears.
            self.close()
        except Exception:  # noqa: BLE001 — destructors must not throw
            LOG.exception("Error during PinnedHostMemory.__del__ cleanup")


__all__ = ["PinnedHostMemory"]
