"""Unit + GPU tests for the ProTrain hardware microbenchmarks.

Covers ``measure_cpu_adam`` and ``measure_gpu_adam`` (§3.2 calibration of
``cost/runtime.py``'s optimizer-step accounting) and the ``HardwareProfile``
default-field contract.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.profiler.hw_bench import (
    measure_cpu_adam,
    measure_gpu_adam,
)
from axolotl.integrations.protrain.types import HardwareProfile


def test_hardware_profile_adam_fields_default_zero():
    """Old trace caches that pickle without the new Adam fields must still
    deserialize — the dataclass default handles that via ``= 0.0``. The
    cost model reads 0.0 and falls back to the hardcoded prior."""
    hw = HardwareProfile(
        gpu_sku="synthetic",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
    )
    assert hw.cpu_adam_bytes_per_sec == 0.0
    assert hw.gpu_adam_bytes_per_sec == 0.0


@pytest.mark.gpu
def test_measure_cpu_adam_returns_sensible_rate():
    """Measured CPU-Adam throughput must be in a plausible DRAM-BW range.

    Allows 0.0 as a valid answer — DeepSpeedCPUAdam requires a matching
    CUDA toolchain to JIT-compile the C++ op, and dev rigs frequently lack
    one. When it DOES compile, observed rates span ~200 MB/s
    (ancient Xeon) to >100 GB/s (modern Threadripper / EPYC + DDR5 +
    cached working set). The upper bound is a unit-error / runaway-value
    guard, not a hardware ceiling — keep it loose enough to accommodate
    high-end DRAM-channel-count CPUs.
    """
    rate = measure_cpu_adam(n_params=2_000_000, n_iters=3)
    if rate == 0.0:
        # DeepSpeedCPUAdam unavailable — the fallback path is exercised
        # by test_estimate_runtime_falls_back_when_adam_bps_zero.
        pytest.skip("DeepSpeedCPUAdam unavailable on this host")
    assert rate >= 100e6, f"CPU Adam rate {rate:.2e} B/s is implausibly low"
    # 1 TB/s upper bound — catches unit errors (e.g. confusing GB with MB)
    # without rejecting modern high-channel-count CPU rigs that genuinely
    # hit 100-200 GB/s on cached benchmarks.
    assert rate <= 1e12, f"CPU Adam rate {rate:.2e} B/s is implausibly high"


def test_measure_cpu_adam_restores_global_del_attribute():
    """The ``__del__`` monkey-patch installed inside ``measure_cpu_adam``
    must NOT leak onto ``DeepSpeedCPUAdam`` after the function returns.

    Production training constructs DeepSpeedCPUAdam via the
    ``CpuFusedAdamAdapter`` AFTER the profiler runs its hardware
    microbenchmarks. If the patched ``__del__`` survived past
    ``measure_cpu_adam`` it would silently swallow legitimate finaliser
    errors in production training optimizers, masking real bugs.

    Verified by snapshotting ``DeepSpeedCPUAdam.__dict__["__del__"]``
    before and after the call (identity comparison — the original is a
    function object defined in DeepSpeed; our patched ``_safe_del`` is a
    fresh closure created on each call).

    Skips if DeepSpeedCPUAdam is unavailable on this host — the warning
    path inside ``measure_cpu_adam`` returns before any monkey-patch
    happens, so there is nothing to verify.
    """
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
    except Exception:  # noqa: BLE001 - import OR JIT compile failure
        pytest.skip("DeepSpeedCPUAdam unavailable on this host")

    sentinel = object()
    pre = DeepSpeedCPUAdam.__dict__.get("__del__", sentinel)

    # Drive the function. We tolerate either a measured rate OR 0.0
    # (constructor failure) — both paths must restore __del__.
    measure_cpu_adam(n_params=1000, n_iters=2)

    post = DeepSpeedCPUAdam.__dict__.get("__del__", sentinel)
    assert post is pre, (
        "measure_cpu_adam left DeepSpeedCPUAdam.__del__ in a different state "
        f"than it found it (was={pre!r}, now={post!r}). The global monkey-"
        "patch leaked past the function boundary, which would silently "
        "swallow finaliser errors in production training optimizers."
    )


@pytest.mark.gpu
def test_measure_gpu_adam_returns_sensible_rate(gpu_device):
    """Measured GPU-Adam throughput must be in a plausible HBM-BW range.

    3090 HBM tops out around 900 GB/s; fused Adam reads/writes ~20 B/param
    in a single kernel call, so sustained rates of 100 GB/s - 2 TB/s are
    expected (the latter only if the kernel is cache-amplified). We
    accept a wide range to avoid flakes on noisy shared hosts, and fall
    back to 0 only if the CUDA context collapses entirely.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rate = measure_gpu_adam(device_idx=gpu_device, n_params=2_000_000, n_iters=3)
    if rate == 0.0:
        pytest.skip("No GPU Adam implementation constructible on this host")
    assert rate >= 10e9, f"GPU Adam rate {rate:.2e} B/s is implausibly low"
    assert rate <= 10e12, f"GPU Adam rate {rate:.2e} B/s is implausibly high"
