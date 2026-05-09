"""Hierarchical chunk management subpackage (ProTrain §3.1.1, Appendix B).

Owns: flattening model states into fixed-size chunks, the persistent vs.
non-persistent split, pre-allocated chunk buffer pool, precise-size pinned
host memory, and the CPU/GPU FusedAdam adapters.

Paper references: MLSys 2026 (arXiv 2406.08334) §3.1.1 and §5, Appendix B.1-B.2.
"""

from __future__ import annotations

from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
from axolotl.integrations.protrain.chunk.layout import build_layout
from axolotl.integrations.protrain.chunk.manager import ChunkManager
from axolotl.integrations.protrain.chunk.optim import (
    CpuFusedAdamAdapter,
    GpuAdamW8bitAdapter,
    GpuFusedAdamAdapter,
)
from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
from axolotl.integrations.protrain.chunk.sizing import pick_S_chunk

__all__ = [
    "BufferPool",
    "ChunkManager",
    "CpuFusedAdamAdapter",
    "GpuAdamW8bitAdapter",
    "GpuFusedAdamAdapter",
    "PinnedHostMemory",
    "build_layout",
    "pick_S_chunk",
]
