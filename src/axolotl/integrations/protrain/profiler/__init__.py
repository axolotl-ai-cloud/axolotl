"""ProTrain memory-aware profiler subpackage."""

from __future__ import annotations

from axolotl.integrations.protrain.profiler.batch_factory import (
    build_batch,
    detect_task_type,
    register_factory,
)
from axolotl.integrations.protrain.profiler.cache import (
    ProfilerCacheKey,
    load_cached_trace,
    save_cached_trace,
)
from axolotl.integrations.protrain.profiler.hw_bench import (
    measure_compute_rate,
    measure_cpu_adam,
    measure_gpu_adam,
    measure_nccl,
    measure_pcie,
)
from axolotl.integrations.protrain.profiler.trace import run_trace
from axolotl.integrations.protrain.types import ProfilerTrace


def reconstruct_peak_bytes(trace: ProfilerTrace) -> int:
    """Simplified peak = model_state + sum(activations) + max(intra) + max(inter)."""
    activations = sum(trace.activation_sizes.values())
    intra = max(0, max(trace.intra_op_delta.values(), default=0))
    inter = max(0, max(trace.inter_op_delta.values(), default=0))
    return int(trace.model_state_bytes + activations + intra + inter)


__all__ = [
    "ProfilerCacheKey",
    "build_batch",
    "detect_task_type",
    "load_cached_trace",
    "measure_compute_rate",
    "measure_cpu_adam",
    "measure_gpu_adam",
    "measure_nccl",
    "measure_pcie",
    "reconstruct_peak_bytes",
    "register_factory",
    "run_trace",
    "save_cached_trace",
]
