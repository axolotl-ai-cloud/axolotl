"""Coordination between publisher ranks, on a gloo group of its own.

Publishing runs on background threads while training issues NCCL collectives on
the default group. Collectives on one group from two threads can interleave and
hang, so every publisher collective goes through a gloo group created once at
init (a collective itself, so every rank creates it from the training thread)
and used only by the publisher thread afterwards.
"""

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


class LocalComm:
    """A single-rank stand-in with the same interface."""

    rank = 0
    world = 1

    def all_gather_object(self, value: Any) -> list[Any]:
        return [value]

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        return value

    def exchange(
        self,
        sends: Sequence[tuple[int, int, torch.Tensor]],
        recvs: Sequence[tuple[int, int, torch.Tensor]],
    ) -> None:
        if sends or recvs:
            raise RuntimeError("A single rank has nothing to exchange")


class GlooComm(LocalComm):
    """Publisher collectives over a dedicated gloo process group."""

    def __init__(self, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self._global = [dist.get_global_rank(group, r) for r in range(self.world)]

    @classmethod
    def create(cls, timeout: timedelta = timedelta(hours=2)) -> "LocalComm":
        """Create the group on every rank; a non-distributed run gets ``LocalComm``."""
        if not (dist.is_available() and dist.is_initialized()):
            return LocalComm()
        if dist.get_world_size() == 1:
            return LocalComm()
        return cls(dist.new_group(backend="gloo", timeout=timeout))

    def all_gather_object(self, value):
        out: list[Any] = [None] * self.world
        dist.all_gather_object(out, value, group=self.group)
        return out

    def broadcast_object(self, value, src=0):
        box = [value]
        dist.broadcast_object_list(box, src=self._global[src], group=self.group)
        return box[0]

    def exchange(self, sends, recvs):
        """Point-to-point copies; tensors travel as raw bytes so any dtype works."""
        handles = []
        for dst, tag, tensor in sends:
            payload = tensor.contiguous().reshape(-1).view(torch.uint8)
            handles.append(
                dist.isend(payload, self._global[dst], group=self.group, tag=tag)
            )
        targets = []
        for src, tag, tensor in recvs:
            payload = torch.empty(
                tensor.numel() * tensor.element_size(), dtype=torch.uint8
            )
            handles.append(
                dist.irecv(payload, self._global[src], group=self.group, tag=tag)
            )
            targets.append((tensor, payload))
        for handle in handles:
            handle.wait()
        for tensor, payload in targets:
            tensor.reshape(-1).view(torch.uint8).copy_(payload)
