"""Single-stream memory allocation context: pins allocations to one CUDA stream's heap."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


class SingleStreamAllocator:
    """Context manager pinning allocations to one CUDA stream (default by default)."""

    def __init__(self, stream: "torch.cuda.Stream | None" = None) -> None:
        # Lazy torch import for CPU-only / docs lanes.
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
        """Block until all ops on self.stream complete; no-op without CUDA."""
        if self.stream is None:
            return
        self.stream.synchronize()


__all__ = ["SingleStreamAllocator"]
