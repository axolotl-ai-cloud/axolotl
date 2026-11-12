"""Copy the live LoRA slices into reused host buffers after an optimizer step.

Each slot is one set of host buffers (pinned when the source is on a GPU),
allocated on first use and reused afterwards. On CUDA the copy runs on a side
stream after the optimizer step's kernels; the next optimizer step waits on its
event on the GPU, so the training thread never blocks on the copy, and the
publisher thread synchronizes on the same event before reading the buffers.
"""

from collections.abc import Sequence

import torch


class LoRAStager:
    def __init__(self, sources: Sequence[torch.Tensor], slots: int = 2):
        self.sources = list(sources)
        self.cuda = any(source.is_cuda for source in self.sources)
        self.buffers: list[list[torch.Tensor] | None] = [None] * slots
        self.events: list = [None] * slots
        self.stream = torch.cuda.Stream() if self.cuda else None
        self._pending = None

    @property
    def nbytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.sources)

    def _allocate(self, slot: int) -> list[torch.Tensor]:
        buffers = self.buffers[slot]
        if buffers is None:
            buffers = self.buffers[slot] = [
                torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=source.is_cuda,
                )
                for source in self.sources
            ]
        return buffers

    @torch.no_grad()
    def stage(self, slot: int) -> None:
        """Start copying every source into ``slot`` (training thread)."""
        buffers = self._allocate(slot)
        if self.stream is None:
            for buffer, source in zip(buffers, self.sources, strict=True):
                buffer.copy_(source)
            self.events[slot] = None
            return
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for buffer, source in zip(buffers, self.sources, strict=True):
                buffer.copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.events[slot] = event
        self._pending = event

    def wait_training(self) -> None:
        """Make the next parameter update wait for the copy, without blocking the host."""
        if self._pending is not None:
            torch.cuda.current_stream().wait_event(self._pending)
            self._pending = None

    def ready(self, slot: int) -> list[torch.Tensor]:
        """The staged buffers of ``slot`` once the copy has finished (publisher thread)."""
        event = self.events[slot]
        if event is not None:
            event.synchronize()
        buffers = self.buffers[slot]
        if buffers is None:
            raise RuntimeError(f"Slot {slot} was never staged")
        return buffers
