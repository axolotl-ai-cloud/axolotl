"""A process-wide IEEE override for fp32 cuBLAS matmuls, and the precision outside it."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import torch

_LOCK = threading.Lock()
_DEPTH = 0
_PREVIOUS: str | None = None


@contextmanager
def ieee_fp32_matmul():
    """Run fp32 cuBLAS matmuls in IEEE regardless of ``tf32``.

    The merge-aware snap kernel accumulates the LoRA delta in IEEE fp32, so every
    other producer of the merged weight (the unfused training path, the merge
    writer, per-step publishing) must too, or TF32 rounding moves elements across
    NVFP4 code boundaries. The flag is process-global: it is refcounted across
    threads so the last exit restores it, and other threads' fp32 matmuls also
    run in IEEE meanwhile.
    """
    global _DEPTH, _PREVIOUS  # noqa: PLW0603
    matmul = torch.backends.cuda.matmul
    with _LOCK:
        if _DEPTH == 0:
            _PREVIOUS = matmul.fp32_precision
            if _PREVIOUS != "ieee":
                matmul.fp32_precision = "ieee"
        _DEPTH += 1
    try:
        yield
    finally:
        with _LOCK:
            _DEPTH -= 1
            if _DEPTH == 0 and _PREVIOUS != "ieee":
                matmul.fp32_precision = _PREVIOUS


def configured_cuda_matmul_precision() -> str:
    """``torch.backends.cuda.matmul.fp32_precision`` as configured, not the
    ``ieee_fp32_matmul`` override another thread may hold right now."""
    with _LOCK:
        if _DEPTH:
            return str(_PREVIOUS)
        return str(torch.backends.cuda.matmul.fp32_precision)
