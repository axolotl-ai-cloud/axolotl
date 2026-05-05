"""ProTrain memory-aware profiler subpackage (M1).

Public surface: a single-GPU, single-iteration tracer that records intra- and
inter-operator memory deltas, hardware microbenchmarks, and a reusable
on-disk cache.
"""

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
    measure_cpu_adam,
    measure_gpu_adam,
    measure_nccl,
    measure_pcie,
)
from axolotl.integrations.protrain.profiler.trace import run_trace
from axolotl.integrations.protrain.types import ProfilerTrace


def reconstruct_peak_bytes(trace: ProfilerTrace) -> int:
    """SIMPLIFIED peak reconstruction for the M1 accuracy contract.

    Returns

        peak = model_state_bytes
             + sum(activation_sizes.values())
             + max(intra_op_delta.values())
             + max(inter_op_delta.values())

    This is intentionally cruder than the full Eqs. 8-11 from the ProTrain
    paper (per-block retained-vs-checkpoint-vs-swap decisions, alpha=1.10
    fragmentation, bumps at the first op of each CKPT block). The full
    reconstruction lives in ``cost/memory.py:estimate_peak``; this simplified
    version provides a peak estimate that matches ``torch.cuda.max_memory_allocated()``
    within ~10 percent on a tiny model with no optimizations enabled, because
    both numbers track the same physical quantity when every block is NONE.
    """
    activations = sum(trace.activation_sizes.values())
    intra = max(trace.intra_op_delta.values(), default=0)
    inter = max(trace.inter_op_delta.values(), default=0)
    return int(trace.model_state_bytes + activations + intra + inter)


__all__ = [
    "ProfilerCacheKey",
    "build_batch",
    "detect_task_type",
    "load_cached_trace",
    "measure_cpu_adam",
    "measure_gpu_adam",
    "measure_nccl",
    "measure_pcie",
    "reconstruct_peak_bytes",
    "register_factory",
    "run_trace",
    "save_cached_trace",
]
