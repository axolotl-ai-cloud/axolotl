"""Hardware microbenchmarks: PCIe H2D/D2H + NCCL collectives + Adam throughput +
per-SKU compute rate."""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# Serialise the global DeepSpeedCPUAdam.__del__ monkey-patch.
_CPU_ADAM_DEL_PATCH_LOCK = threading.Lock()


@contextmanager
def _patched_deepspeed_cpu_adam_del(deepspeed_cpu_adam_cls: type) -> Iterator[None]:
    """Install a partial-init-safe ``__del__`` on DeepSpeedCPUAdam for the context."""
    with _CPU_ADAM_DEL_PATCH_LOCK:
        sentinel = object()
        original = deepspeed_cpu_adam_cls.__dict__.get("__del__", sentinel)

        def _safe_del(self: object) -> None:
            try:
                if hasattr(self, "ds_opt_adam") and original is not sentinel:
                    original(self)  # type: ignore[misc, operator]
            except Exception:  # noqa: BLE001 - suppress silently; dev-rig safety
                pass

        deepspeed_cpu_adam_cls.__del__ = _safe_del  # type: ignore[attr-defined]
        try:
            yield
        finally:
            # Restore exact prior state: delete override if class lacked __del__, else rebind.
            if original is sentinel:
                try:
                    del deepspeed_cpu_adam_cls.__del__  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            else:
                deepspeed_cpu_adam_cls.__del__ = original  # type: ignore[attr-defined]


# Reference compute rate (TFLOPS, fp16) used to scale per-SKU calibration ratios
# when neither the trace nor the live HardwareProfile reports a measurement.
# 71 TFLOPS is the published RTX 3090 fp16-tensor-core peak (a 3090 Ti is
# nominally ~80 TFLOPS) — sustained throughput measured by ``measure_compute_rate``
# typically lands around 60-65% of peak under the GEMM workload.
DEFAULT_COMPUTE_RATE_TFLOPS: float = 50.0


# Adam bytes/param accounting; conservative 20 B/param (fused kernels lower the actual count).
_ADAM_BYTES_PER_PARAM: int = 20


def measure_pcie(
    device_idx: int = 0,
    n_bytes: int = 256 * 1024 * 1024,
    n_iters: int = 5,
) -> tuple[float, float]:
    """Measure sustained H2D and D2H bandwidth on a single device; returns (h2d_bps, d2h_bps)."""
    if n_iters < 1:
        raise ValueError(f"measure_pcie: n_iters must be >= 1, got {n_iters}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measure_pcie requires CUDA.")

    device = torch.device(f"cuda:{device_idx}")

    # uint8 so n_bytes == numel(); pinned host memory for true async copies.
    host = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(n_bytes, dtype=torch.uint8, device=device)

    # Bind events + record + sync to device_idx under one device guard.
    h2d_times: list[float] = []
    d2h_times: list[float] = []
    with torch.cuda.device(device_idx):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def _time_copy(src, dst) -> float:
            torch.cuda.synchronize(device)
            start.record()
            dst.copy_(src, non_blocking=True)
            end.record()
            torch.cuda.synchronize(device)
            # elapsed_time is in ms
            return start.elapsed_time(end) / 1000.0

        # Warmup + measured iters, H2D
        for i in range(n_iters + 1):
            t = _time_copy(host, gpu)
            if i > 0:
                h2d_times.append(t)

        for i in range(n_iters + 1):
            t = _time_copy(gpu, host)
            if i > 0:
                d2h_times.append(t)

    h2d_bps = n_bytes / (sum(h2d_times) / len(h2d_times))
    d2h_bps = n_bytes / (sum(d2h_times) / len(d2h_times))

    LOG.debug(
        "measure_pcie device=%d h2d=%.2f GB/s d2h=%.2f GB/s",
        device_idx,
        h2d_bps / 1e9,
        d2h_bps / 1e9,
    )
    return h2d_bps, d2h_bps


def measure_cpu_adam(n_params: int = 10_000_000, n_iters: int = 10) -> float:
    """Return bytes/sec throughput of CPU Adam; 0.0 on import/compile failure."""
    if n_iters < 1:
        raise ValueError(f"measure_cpu_adam: n_iters must be >= 1, got {n_iters}")

    try:
        from deepspeed.ops.adam import (
            DeepSpeedCPUAdam,  # type: ignore[import-not-found]
        )
    except Exception as exc:  # noqa: BLE001 - import OR compile failure
        LOG.warning(
            "measure_cpu_adam: DeepSpeedCPUAdam unavailable (%s); "
            "returning 0.0 so the runtime cost model falls back to a "
            "hardcoded prior",
            exc,
        )
        return 0.0

    import torch
    from torch import nn

    with _patched_deepspeed_cpu_adam_del(DeepSpeedCPUAdam):
        # fp32 master + 2x momenta allocated internally on first step.
        param = nn.Parameter(
            torch.randn(n_params, dtype=torch.float16, device="cpu"),
            requires_grad=True,
        )
        param.grad = torch.randn(n_params, dtype=torch.float16, device="cpu")

        try:
            optim = DeepSpeedCPUAdam([param], lr=1e-4)
        except Exception as exc:  # noqa: BLE001 - CUDA toolchain mismatch etc.
            LOG.warning(
                "measure_cpu_adam: DeepSpeedCPUAdam constructor failed (%s); returning 0.0",
                repr(exc),
            )
            # Drop traceback so frame locals don't pin autograd tensors.
            exc.__traceback__ = None
            del exc, param
            return 0.0

        # Warmup — first step allocates optimizer state and JITs the kernel.
        try:
            optim.step()
        except Exception as exc:  # noqa: BLE001 - defensive
            LOG.warning("measure_cpu_adam: warmup step failed (%s); returning 0.0", exc)
            return 0.0

        iter_s: list[float] = []
        for _ in range(n_iters):
            # Re-populate grad each iter for steady-state kernel cost.
            param.grad = torch.randn(n_params, dtype=torch.float16, device="cpu")
            t0 = time.perf_counter()
            optim.step()
            iter_s.append(time.perf_counter() - t0)

        median_iter = statistics.median(iter_s)
        if median_iter <= 0:
            bps = 0.0
        else:
            bytes_processed = n_params * _ADAM_BYTES_PER_PARAM
            bps = bytes_processed / median_iter
            LOG.debug(
                "measure_cpu_adam n_params=%d median_iter=%.4fs throughput=%.2f GB/s",
                n_params,
                median_iter,
                bps / 1e9,
            )
        # Explicit cleanup; skip gc.collect to avoid pytest unraisable-warning interference.
        try:
            optim.zero_grad(set_to_none=True)
            optim.state.clear()
        except Exception:  # noqa: BLE001 - defensive
            pass
        del optim, param
        return float(bps)


def measure_gpu_adam(
    device_idx: int = 0, n_params: int = 5_000_000, n_iters: int = 10
) -> float:
    """Return bytes/sec throughput of GPU Adam; 0.0 when no backend works."""
    if n_iters < 1:
        raise ValueError(f"measure_gpu_adam: n_iters must be >= 1, got {n_iters}")

    import torch
    from torch import nn

    if not torch.cuda.is_available():
        LOG.warning("measure_gpu_adam: CUDA unavailable; returning 0.0")
        return 0.0

    device = torch.device(f"cuda:{device_idx}")

    param = nn.Parameter(
        torch.randn(n_params, dtype=torch.float16, device=device),
        requires_grad=True,
    )
    param.grad = torch.randn(n_params, dtype=torch.float16, device=device)

    # Backend selected only if BOTH construction AND warmup succeed.
    def _try_apex() -> torch.optim.Optimizer:
        from apex.optimizers import FusedAdam  # type: ignore[import-not-found]

        return FusedAdam([param], lr=1e-4)

    def _try_torch_fused() -> torch.optim.Optimizer:
        return torch.optim.AdamW([param], lr=1e-4, fused=True)

    def _try_torch_stock() -> torch.optim.Optimizer:
        return torch.optim.AdamW([param], lr=1e-4)

    candidates = [
        ("apex.FusedAdam", _try_apex),
        ("torch.optim.AdamW(fused=True)", _try_torch_fused),
        ("torch.optim.AdamW", _try_torch_stock),
    ]

    optim = None
    backend = ""
    for name, build in candidates:
        try:
            candidate = build()
            # Warmup + JIT — must run on the same backend we plan to time.
            candidate.step()
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 - any failure → next backend
            LOG.debug(
                "measure_gpu_adam: backend=%s failed (%s); trying next", name, exc
            )
            # Discard half-initialized optimizer so next candidate starts clean.
            try:
                del candidate  # type: ignore[possibly-unused-variable]
            except UnboundLocalError:
                pass
            continue
        optim = candidate
        backend = name
        break

    if optim is None:
        LOG.warning(
            "measure_gpu_adam: no Adam backend succeeded (apex / fused AdamW / "
            "stock AdamW all failed); returning 0.0"
        )
        return 0.0

    LOG.debug("measure_gpu_adam: backend=%s", backend)

    iter_s: list[float] = []
    # Bind events + record + synchronize to ``device_idx`` so they don't
    # latch onto a stale ``current_device()`` under multi-GPU / masking.
    with torch.cuda.device(device_idx):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(n_iters):
            # Re-issue a fresh grad each iter. Keep it simple — copy in place
            # so we don't thrash the allocator.
            param.grad.copy_(torch.randn_like(param.grad))
            torch.cuda.synchronize(device)
            start.record()
            optim.step()
            end.record()
            torch.cuda.synchronize(device)
            iter_s.append(start.elapsed_time(end) / 1000.0)

    median_iter = statistics.median(iter_s)
    bytes_processed = n_params * _ADAM_BYTES_PER_PARAM
    bps = bytes_processed / median_iter if median_iter > 0 else 0.0
    LOG.debug(
        "measure_gpu_adam backend=%s n_params=%d median_iter=%.4fs throughput=%.2f GB/s",
        backend,
        n_params,
        median_iter,
        bps / 1e9,
    )
    # Release the synthetic param + optimizer state before returning.
    # Fused AdamW holds references to optim-state tensors in ``optim.state``
    # and sometimes via CUDA graph caches, so a plain ``del`` isn't enough.
    # We explicitly clear the state dict and zero out ``param.data`` so the
    # No empty_cache; would force the traced forward to re-reserve memory.
    try:
        optim.zero_grad(set_to_none=True)
        optim.state.clear()
        optim.param_groups.clear()
    except Exception:  # noqa: BLE001 - defensive, no behavior change
        pass
    param.grad = None
    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    del optim, param
    torch.cuda.synchronize(device)
    return float(bps)


# NCCL benchmark payload sizes spanning ProTrain's S_chunk range + small-collective regime.
NCCL_PAYLOAD_SIZES_BYTES: tuple[int, ...] = (
    1 << 20,  # 1 MiB
    4 << 20,  # 4 MiB
    16 << 20,  # 16 MiB
    64 << 20,  # 64 MiB
    256 << 20,  # 256 MiB
)


def measure_nccl(
    world_size: int,
    *,
    payload_sizes_bytes: tuple[int, ...] = NCCL_PAYLOAD_SIZES_BYTES,
    n_iters: int = 8,
    n_warmup: int = 2,
) -> tuple[dict[int, float], dict[int, float]]:
    """Measure NCCL gather + reduce latencies per payload size."""
    if n_iters < 1:
        raise ValueError(f"measure_nccl: n_iters must be >= 1, got {n_iters}")
    if n_warmup < 0:
        raise ValueError(f"measure_nccl: n_warmup must be >= 0, got {n_warmup}")

    import torch
    import torch.distributed as dist

    # Single-rank fast path: validate against runtime world size first to catch misconfig.
    if world_size == 1:
        if not dist.is_available() or not dist.is_initialized():
            return ({}, {})
        runtime_world = dist.get_world_size()
        if runtime_world == 1:
            return ({}, {})
        raise RuntimeError(
            f"measure_nccl: caller passed world_size=1 but "
            f"torch.distributed reports world_size={runtime_world}. "
            "Either pass the actual world size or tear down the "
            "distributed group before calling for the single-rank fast path."
        )

    if not dist.is_available():
        raise RuntimeError(
            "measure_nccl: torch.distributed unavailable — rebuild PyTorch "
            "with NCCL/Gloo support to use multi-rank profiling."
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "measure_nccl: torch.distributed not initialized. Run under "
            "torchrun, or use scripts/protrain/measure_nccl.py which "
            "bootstraps the rendezvous itself. "
            f"Caller passed world_size={world_size}."
        )
    actual_world = dist.get_world_size()
    if actual_world != world_size:
        raise RuntimeError(
            f"measure_nccl: caller passed world_size={world_size} but "
            f"torch.distributed reports world_size={actual_world}. Check "
            "your launcher / environment for a misconfiguration."
        )

    rank = dist.get_rank()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "measure_nccl requires CUDA — NCCL collectives need GPU tensors."
        )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # Extract ordinal for device guards against stale current_device under masking.
    device_idx = device.index if device.index is not None else 0

    gather_table: dict[int, float] = {}
    reduce_table: dict[int, float] = {}

    # Pre-collective barrier surfaces communicator asymmetry as a debuggable hang.
    try:
        dist.barrier(device_ids=[device_idx])
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "measure_nccl: pre-collective dist.barrier() failed — your ranks "
            "likely have asymmetric NCCL communicator config. Set "
            "TORCH_DISTRIBUTED_DEBUG=DETAIL and re-run to inspect."
        ) from exc

    for payload_bytes in payload_sizes_bytes:
        # Key by actually-benchmarked bytes so non-divisible payloads don't mis-label.
        element_size = 4  # float32
        elements_per_shard = max(1, (payload_bytes // world_size) // element_size)
        actual_payload_bytes = elements_per_shard * world_size * element_size
        shard = torch.zeros(elements_per_shard, dtype=torch.float32, device=device)
        gathered = torch.zeros(
            elements_per_shard * world_size,
            dtype=torch.float32,
            device=device,
        )

        # Warmup
        for _ in range(n_warmup):
            dist.all_gather_into_tensor(gathered, shard)
        torch.cuda.synchronize(device)

        # Timed under one device guard for amortised cost.
        gather_times: list[float] = []
        with torch.cuda.device(device_idx):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iters):
                start.record()
                dist.all_gather_into_tensor(gathered, shard)
                end.record()
                torch.cuda.synchronize(device)
                gather_times.append(start.elapsed_time(end) / 1000.0)
        gather_table[actual_payload_bytes] = statistics.median(gather_times)

        # reduce_scatter_tensor: full payload in, one shard out per rank.
        full_payload = torch.zeros(
            elements_per_shard * world_size,
            dtype=torch.float32,
            device=device,
        )
        reduced = torch.zeros(elements_per_shard, dtype=torch.float32, device=device)

        # op=AVG mirrors production; NCCL AVG = SUM + post-divide (slightly costlier).
        for _ in range(n_warmup):
            dist.reduce_scatter_tensor(reduced, full_payload, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize(device)

        # Timed under one device guard for amortised cost.
        reduce_times: list[float] = []
        with torch.cuda.device(device_idx):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iters):
                start.record()
                dist.reduce_scatter_tensor(reduced, full_payload, op=dist.ReduceOp.AVG)
                end.record()
                torch.cuda.synchronize(device)
                reduce_times.append(start.elapsed_time(end) / 1000.0)
        reduce_table[actual_payload_bytes] = statistics.median(reduce_times)

        del shard, gathered, full_payload, reduced
        # empty_cache before next payload to avoid fragmenting the pool at large sizes.
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - defensive, no behavior change
                pass

        if rank == 0:
            LOG.debug(
                "measure_nccl payload=%dMiB gather=%.3fms reduce=%.3fms "
                "(world=%d, %d iters)",
                payload_bytes >> 20,
                gather_table[payload_bytes] * 1000,
                reduce_table[payload_bytes] * 1000,
                world_size,
                n_iters,
            )

    return gather_table, reduce_table


def measure_compute_rate(
    device_idx: int = 0,
    *,
    matrix_size: int = 4096,
    n_iters: int = 10,
    n_warmup: int = 3,
) -> float:
    """Return sustained fp16 compute throughput in TFLOPS for ``device_idx``."""
    if n_iters < 1:
        raise ValueError(f"measure_compute_rate: n_iters must be >= 1, got {n_iters}")
    if n_warmup < 0:
        raise ValueError(f"measure_compute_rate: n_warmup must be >= 0, got {n_warmup}")

    import torch

    if not torch.cuda.is_available():
        LOG.warning("measure_compute_rate: CUDA unavailable; returning 0.0")
        return 0.0

    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    # Preallocate output so the hot loop has no allocator overhead.
    c = torch.empty(matrix_size, matrix_size, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(n_warmup):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize(device)

    # Timed under device guard so events don't latch onto stale current_device.
    iter_s: list[float] = []
    with torch.cuda.device(device_idx):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(n_iters):
            start.record()
            torch.matmul(a, b, out=c)
            end.record()
            torch.cuda.synchronize(device)
            iter_s.append(start.elapsed_time(end) / 1000.0)
    median_iter = statistics.median(iter_s)

    # Square-matmul FLOP count: 2 * N^3.
    flops_per_iter = 2.0 * (matrix_size**3)
    tflops = flops_per_iter / median_iter / 1e12

    LOG.debug(
        "measure_compute_rate device=%d N=%d median_iter=%.4fs throughput=%.2f TFLOPS",
        device_idx,
        matrix_size,
        median_iter,
        tflops,
    )

    # Cleanup
    del a, b, c
    torch.cuda.synchronize(device)
    return float(tflops)


__all__ = [
    "measure_pcie",
    "measure_nccl",
    "measure_cpu_adam",
    "measure_gpu_adam",
    "measure_compute_rate",
    "NCCL_PAYLOAD_SIZES_BYTES",
    "DEFAULT_COMPUTE_RATE_TFLOPS",
]
