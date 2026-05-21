"""Intra- and inter-operator memory delta capture via torch.cuda.memory_stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


def intra_op_delta(before_bytes: int, peak_bytes: int) -> int:
    """Transient bytes inside op: max(0, peak_during - allocated_before)."""
    return max(0, peak_bytes - before_bytes)


def inter_op_delta(prev_end_bytes: int, curr_peak_bytes: int) -> int:
    """Bytes allocated between hooks; covers the ~17% invisible peak hooks miss."""
    return max(0, curr_peak_bytes - prev_end_bytes)


@dataclass
class MemorySnapshot:
    """Lightweight snapshot of the CUDA allocator state at one point in time."""

    allocated_bytes: int
    peak_allocated_bytes: int


class MemoryDeltaTracker:
    """Wraps torch.cuda.memory_stats; reset → snapshot pre/post for hooks."""

    def __init__(self, device: "torch.device | str | int | None" = None) -> None:
        """Bind to device; seed inter-op baseline as unset (None sentinel)."""
        import torch

        self._torch = torch
        self._device = device
        # None sentinel: first delta_since_last returns 0 and sets baseline.
        self._last_end_bytes: int | None = None

    # ---- allocator interface --------------------------------------------

    def _stats(self) -> dict:
        # Guard CPU-only hosts and non-CUDA device strings; empty dict → snapshot zeros via .get.
        if self._device is not None and not isinstance(self._device, int):
            if self._torch.device(self._device).type != "cuda":
                return {}
        if not self._torch.cuda.is_available():
            return {}
        return self._torch.cuda.memory_stats(self._device)

    def reset(self) -> None:
        """Reset peak tracker on device; no-op on CPU-only or non-CUDA device."""
        if self._torch.cuda.is_available() and (
            self._device is None
            or isinstance(self._device, int)
            or self._torch.device(self._device).type == "cuda"
        ):
            self._torch.cuda.reset_peak_memory_stats(self._device)

    def snapshot(self) -> MemorySnapshot:
        """Return current allocator state (allocated + peak-since-last-reset)."""
        stats = self._stats()
        allocated = int(stats.get("allocated_bytes.all.current", 0))
        peak = int(stats.get("allocated_bytes.all.peak", allocated))
        return MemorySnapshot(allocated_bytes=allocated, peak_allocated_bytes=peak)

    def delta_since_last(self) -> int:
        """Bytes allocated since last call; first call returns 0 and sets baseline."""
        snap = self.snapshot()
        current = snap.allocated_bytes
        if self._last_end_bytes is None:
            self._last_end_bytes = current
            self.reset()
            return 0
        delta = max(0, snap.peak_allocated_bytes - self._last_end_bytes)
        self._last_end_bytes = current
        # Reset peak window so next interval starts fresh.
        self.reset()
        return delta

    def mark_end(self, end_bytes: int) -> None:
        """Record allocated_bytes at op end; resets peak window for the next interval."""
        self._last_end_bytes = end_bytes
        self.reset()

    @property
    def last_end_bytes(self) -> int:
        return 0 if self._last_end_bytes is None else self._last_end_bytes


__all__ = [
    "MemoryDeltaTracker",
    "MemorySnapshot",
    "inter_op_delta",
    "intra_op_delta",
]
