"""Pinned-RAM activation pool for SWAP block path; one PinnedHostMemory backs n_slot uint8 slots."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


# Saved-tensors per block (residual + Q/K/V/scores + FFN intermediates).
DEFAULT_SLOTS_PER_BLOCK: int = 8

# close() drain timeout for in-flight borrows.
DEFAULT_CLOSE_DRAIN_TIMEOUT_S: float = 30.0


class ActivationSwapPool:
    """Fixed-size pinned-host slot pool: n_swap * slots_per_block * prefetch_depth slots.

    Notes
    -----
    The pool is **stream-agnostic** — copies onto/from slots happen on
    the SWAP wrapper's chosen stream (typically the scheduler's
    ``_swap_stream``). Slot ownership is tracked by Python-side ID
    only; CUDA never sees the pool's free-list state. Callers MUST
    synchronize the swap stream with their consumer before
    """

    def __init__(
        self,
        n_swap: int,
        slot_bytes: int,
        prefetch_depth: int = 2,
        slots_per_block: int = DEFAULT_SLOTS_PER_BLOCK,
    ) -> None:
        """Allocate the backing pinned region and the free-slot LIFO."""
        if n_swap < 1:
            raise ValueError(f"n_swap must be >= 1, got {n_swap}")
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
        if prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth}")
        if slots_per_block < 1:
            raise ValueError(f"slots_per_block must be >= 1, got {slots_per_block}")

        self.n_swap = int(n_swap)
        self.slot_bytes = int(slot_bytes)
        self.prefetch_depth = int(prefetch_depth)
        self.slots_per_block = int(slots_per_block)
        self.n_slot = self.n_swap * self.slots_per_block * self.prefetch_depth

        # Backing pinned-host region split into n_slot equal slots.
        self._pinned = PinnedHostMemory(n_buffer=self.n_slot, S_chunk=self.slot_bytes)
        self._closed = False
        # Blocks new acquires during close()'s drain window.
        self._closing = False
        # LIFO free-list of slot indices.
        self._free: list[int] = list(range(self.n_slot))
        self._inflight: int = 0
        # Lock against autograd worker / main thread races on free + inflight.
        self._lock = threading.Lock()

        LOG.debug(
            "ActivationSwapPool: n_swap=%d slot_bytes=%d prefetch_depth=%d "
            "slots_per_block=%d n_slot=%d total_bytes=%d precise=%s",
            self.n_swap,
            self.slot_bytes,
            self.prefetch_depth,
            self.slots_per_block,
            self.n_slot,
            self.n_slot * self.slot_bytes,
            self._pinned.is_precise_size,
        )

    def acquire(self) -> tuple[int, "torch.Tensor"]:
        """Reserve a slot; return (slot_id, pinned uint8 view)."""
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("ActivationSwapPool is closed")
            if not self._free:
                raise RuntimeError(
                    f"ActivationSwapPool exhausted (n_slot={self.n_slot}, "
                    f"in-flight={self._inflight}); increase prefetch_depth or "
                    "verify the SWAP wrapper releases slots after backward."
                )
            slot_id = self._free.pop()
            self._inflight += 1
            # PinnedHostMemory.buffer() mutates _live_borrows; hold lock + roll back on raise.
            try:
                view = self._pinned.buffer(slot_id)
            except BaseException:
                self._inflight -= 1
                self._free.append(slot_id)
                raise
        return slot_id, view

    def release(self, slot_id: int) -> None:
        """Return slot_id to free list; pool does NOT issue stream syncs."""
        with self._lock:
            # release() proceeds during _closing so close()'s drain can converge.
            if self._closed:
                return
            if not 0 <= slot_id < self.n_slot:
                LOG.warning(
                    "ActivationSwapPool.release: slot_id %d out of range [0, %d); ignored",
                    slot_id,
                    self.n_slot,
                )
                return
            if slot_id in self._free:
                LOG.warning(
                    "ActivationSwapPool.release: slot %d already free; double-release",
                    slot_id,
                )
                return
            # release_buffer first; if it raises, slot stays in inflight (no double-acquire).
            self._pinned.release_buffer(slot_id)
            self._free.append(slot_id)
            self._inflight -= 1

    @property
    def total_bytes(self) -> int:
        """Total pinned-host bytes held by the pool."""
        return self.n_slot * self.slot_bytes

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return self._inflight

    def close(
        self,
        drain_timeout: float = DEFAULT_CLOSE_DRAIN_TIMEOUT_S,
        poll_interval: float = 0.01,
    ) -> None:
        """Free the pinned region. Three-phase: closing flag → drain inflight → free.  Idempotent."""
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError(
                    "ActivationSwapPool.close: close in progress on another thread"
                )
            self._closing = True
        try:
            # Drain in-flight borrows; release() converges _inflight to 0.
            deadline = time.monotonic() + max(0.0, float(drain_timeout))
            while True:
                with self._lock:
                    inflight = self._inflight
                if inflight == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"ActivationSwapPool.close: timed out after "
                        f"{drain_timeout:.3f}s waiting for {inflight} "
                        "in-flight slot(s) to drain. Caller is missing "
                        "release() pairs or the swap stream has not "
                        "synchronized — retry close() after stragglers "
                        "retire."
                    )
                time.sleep(max(0.0, float(poll_interval)))
            # Drain complete; free the pinned region.
            self._pinned.close()
        except BaseException:
            # Roll back _closing so callers can retry.
            with self._lock:
                self._closing = False
            raise
        with self._lock:
            self._closed = True
            self._free.clear()
            self._inflight = 0

    def __del__(self) -> None:  # noqa: D401
        try:
            # Non-blocking cleanup; 30s drain would stall shutdown.
            self.close(drain_timeout=0)
        except Exception:  # noqa: BLE001 — destructor must not throw
            # Surface teardown-time leaks at debug level so they're
            # visible without breaking the destructor's no-throw contract.
            LOG.debug("ActivationSwapPool.__del__ cleanup skipped", exc_info=True)


__all__ = ["ActivationSwapPool", "DEFAULT_SLOTS_PER_BLOCK"]
