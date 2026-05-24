"""ProTrain cost models (M4).

Implements Eqs. 2-11 from the MLSys 2026 paper:

- ``estimate_runtime`` — wall-clock seconds per iteration (Eqs. 2-7).
- ``estimate_peak`` — peak GPU bytes with alpha fragmentation (Eqs. 8-11).
- ``effective_bw`` — PCIe bandwidth derate under SWAP contention (§3.3).

These are pure functions of ``ProfilerTrace`` + ``ChunkLayout`` +
``BlockStrategyMap`` + ``HardwareProfile``; they do not allocate tensors
or require a GPU.
"""

from __future__ import annotations

from axolotl.integrations.protrain.cost.bandwidth import effective_bw
from axolotl.integrations.protrain.cost.memory import (
    ALPHA_FRAGMENTATION,
    ALPHA_FRAGMENTATION_4BIT,
    ALPHA_FRAGMENTATION_4BIT_MODE_A,
    ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT,
    alpha_fragmentation_for_cfg,
    estimate_cpu_footprint,
    estimate_peak,
)
from axolotl.integrations.protrain.cost.runtime import estimate_runtime

__all__ = [
    "ALPHA_FRAGMENTATION",
    "ALPHA_FRAGMENTATION_4BIT",
    "ALPHA_FRAGMENTATION_4BIT_MODE_A",
    "ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT",
    "alpha_fragmentation_for_cfg",
    "effective_bw",
    "estimate_cpu_footprint",
    "estimate_peak",
    "estimate_runtime",
]
