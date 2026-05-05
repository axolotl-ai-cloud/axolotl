"""Intra- and inter-operator memory delta capture via torch.cuda.memory_stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


def intra_op_delta(before_bytes: int, peak_bytes: int) -> int:
    """Transient bytes allocated *inside* an op: ``peak_during - allocated_before``.

    Clamped at zero — a negative delta means the op freed memory before
    allocating (rare) and we treat that as zero transient overhead.
    """
    return max(0, peak_bytes - before_bytes)


def inter_op_delta(prev_end_bytes: int, curr_peak_bytes: int) -> int:
    """Bytes allocated *between* recorded hooks (unhookable ``nn.functional.*`` etc.).

    Paper §3.2 / Appendix A.2: this is the ~17% invisible peak that
    ``torch.profiler`` and naive layer hooks miss.
    """
    return max(0, curr_peak_bytes - prev_end_bytes)


@dataclass
class MemorySnapshot:
    """Lightweight snapshot of the CUDA allocator state at one point in time."""

    allocated_bytes: int
    peak_allocated_bytes: int


class MemoryDeltaTracker:
    """Wraps ``torch.cuda.memory_stats`` so hooks can read/reset without import churn.

    Usage pattern from ``trace.py``:

        tracker = MemoryDeltaTracker(device)
        # pre-forward hook:
        tracker.reset()
        before = tracker.snapshot()
        # post-forward hook:
        after = tracker.snapshot()
        intra = intra_op_delta(before.allocated_bytes, after.peak_allocated_bytes)
    """

    def __init__(self, device: "torch.device | str | int | None" = None) -> None:
        """Bind the tracker to ``device`` and seed the inter-op baseline as unset."""
        # Local import so this module can be parsed in environments without
        # torch installed (e.g. syntax check in CI prep).
        import torch

        self._torch = torch
        self._device = device
        # ``None`` sentinel so the first ``delta_since_last`` call establishes
        # the baseline and returns 0, instead of treating "0 bytes" as the
        # previous end and reporting the entire current allocation as the
        # delta. ``mark_end`` (explicit baseline-set) is unchanged.
        self._last_end_bytes: int | None = None

    # ---- allocator interface --------------------------------------------

    def _stats(self) -> dict:
        # ``torch.cuda.memory_stats`` raises on CPU-only hosts (it's a CUDA-
        # specific API that requires an initialized CUDA context). Guard with
        # ``is_available()`` so callers on CPU-only machines get an empty dict
        # and ``snapshot()`` falls back to zeros via ``.get()`` defaults.
        # Also guard against a non-CUDA device passed on a GPU host: ``int``
        # is always a CUDA device index and ``None`` means "current device",
        # but ``str``/``torch.device`` may resolve to ``"cpu"``/``"mps"`` etc.
        if self._device is not None and not isinstance(self._device, int):
            if self._torch.device(self._device).type != "cuda":
                return {}
        if not self._torch.cuda.is_available():
            return {}
        return self._torch.cuda.memory_stats(self._device)

    def reset(self) -> None:
        """Reset the ``peak_*`` tracker on the device so the next snapshot is local.

        Guarded by ``torch.cuda.is_available()`` so external callers on CPU-only
        hosts get a no-op rather than a CUDA-init error. ``snapshot()`` is
        already safe because ``memory_stats()`` returns an empty dict when CUDA
        is unavailable and ``.get()`` defaults handle the missing keys. Also
        no-op when ``self._device`` resolves to a non-CUDA device on a GPU host
        (``int``/``None`` always pass; ``str``/``torch.device`` are normalized).
        """
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
        """Return bytes allocated since the last ``delta_since_last`` call.

        First call establishes the baseline and returns 0. Intended for the
        inter-op hook slot where the "previous end" is whatever the last
        post-op hook observed.

        Uses ``peak_allocated_bytes`` (not ``allocated_bytes``) for the delta
        so transient spikes that allocate-then-free between hooks are still
        counted — that inter-op transient is exactly what this module exists
        to recover (paper §3.2 / Appendix A.2). The baseline is then advanced
        with the current ``allocated_bytes`` so the next call measures growth
        from the post-op resident set.
        """
        snap = self.snapshot()
        current = snap.allocated_bytes
        if self._last_end_bytes is None:
            self._last_end_bytes = current
            return 0
        delta = max(0, snap.peak_allocated_bytes - self._last_end_bytes)
        self._last_end_bytes = current
        return delta

    def mark_end(self, end_bytes: int) -> None:
        """Record the ``allocated_bytes`` at the end of an op, for inter-op delta."""
        self._last_end_bytes = end_bytes

    @property
    def last_end_bytes(self) -> int:
        return 0 if self._last_end_bytes is None else self._last_end_bytes


__all__ = [
    "MemoryDeltaTracker",
    "MemorySnapshot",
    "inter_op_delta",
    "intra_op_delta",
]
