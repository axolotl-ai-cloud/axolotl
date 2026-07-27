"""Shared guards and launch helpers for the ternary Triton kernels."""

from __future__ import annotations

import torch

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def require_cuda(*tensors: torch.Tensor) -> None:
    """Raise unless Triton is importable and every tensor lives on CUDA.

    Raises:
        RuntimeError: If Triton is missing or a tensor is not on a CUDA device.
    """
    if not HAVE_TRITON:  # pragma: no cover
        raise RuntimeError("triton is required for the fused ternary kernels")
    for tensor in tensors:
        if not tensor.is_cuda:
            raise RuntimeError(
                f"the fused ternary kernels need CUDA tensors, got {tensor.device}"
            )
        if hasattr(tensor, "to_local") or hasattr(
            getattr(tensor, "data", None), "to_local"
        ):
            # a DTensor reports the global numel over local-shard storage, so a
            # kernel sized from it walks off the allocation
            raise RuntimeError(
                f"the fused ternary kernels need plain tensors, got "
                f"{type(tensor).__name__}; pass `to_local()` or `full_tensor()`"
            )


def num_warps_for(block: int) -> int:
    """Return the warp count for an elementwise kernel with `block` elements per program."""
    if block >= 4096:
        return 8
    if block >= 1024:
        return 4
    return 2
