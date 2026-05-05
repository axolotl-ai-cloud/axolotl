"""Hardware microbenchmarks: PCIe H2D/D2H + NCCL collectives + Adam throughput +
per-SKU compute rate."""

from __future__ import annotations

import statistics
import time

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# Reference compute rate (TFLOPS, fp16) used to scale per-SKU calibration ratios
# when neither the trace nor the live HardwareProfile reports a measurement.
# 71 TFLOPS is the published RTX 3090 fp16-tensor-core peak (a 3090 Ti is
# nominally ~80 TFLOPS) — sustained throughput measured by ``measure_compute_rate``
# typically lands around 60-65% of peak under the GEMM workload.
DEFAULT_COMPUTE_RATE_TFLOPS: float = 50.0


# Bytes-per-param accounting used by the Adam microbenchmarks below.
# Breakdown (simplified; see module docstring in cost/runtime.py):
#   fp16 param    : 2 B read + 2 B write = 4 B
#   fp16 grad     : 2 B read             = 2 B
#   fp32 master   : 4 B read + 4 B write = 8 B
#   fp32 momentum : 4 B read + 4 B write = 8 B
#   fp32 variance : 4 B read + 4 B write = 8 B (counted as 2x momentum below)
# Collapsing the two momenta into a single "2x momentum" term and rounding
# to the roofline-style estimate the paper uses lands at ~30 B/param. We
# keep the constant conservative (20 B/param) because DeepSpeedCPUAdam and
# apex FusedAdam both fuse the master+momenta update into a single kernel
# that does fewer round-trips to DRAM than the naive count predicts. The
# MEASURED throughput returned is empirical regardless; this constant only
# determines the units (bytes/sec) we report.
_ADAM_BYTES_PER_PARAM: int = 20


def measure_pcie(
    device_idx: int = 0,
    n_bytes: int = 256 * 1024 * 1024,
    n_iters: int = 5,
) -> tuple[float, float]:
    """Measure sustained H2D and D2H bandwidth on a single device.

    Uses a pinned host tensor and ``torch.cuda.Event`` for timing. Returns
    ``(h2d_bps, d2h_bps)`` in bytes/sec.

    Args:
        device_idx: CUDA device ordinal.
        n_bytes: payload size. 256 MiB is large enough to saturate PCIe 4.0 x16
            on a 3090 (~26 GB/s peak) without blowing up small-device budgets.
        n_iters: repetitions — the first is a warmup and is discarded.
    """
    if n_iters < 1:
        raise ValueError(f"measure_pcie: n_iters must be >= 1, got {n_iters}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measure_pcie requires CUDA.")

    device = torch.device(f"cuda:{device_idx}")

    # uint8 so n_bytes == numel(); pinned host memory for true async copies.
    host = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(n_bytes, dtype=torch.uint8, device=device)

    # Bind the timing events to ``device_idx`` so they record on the
    # right device under CUDA_VISIBLE_DEVICES masking / multi-GPU rigs.
    # ``torch.cuda.Event`` infers its device from the current device at
    # construction time AND ``event.record()`` / ``torch.cuda.synchronize``
    # are device-bound operations — if any of these run with a different
    # default device than the events were created on, the events bind to
    # the wrong stream/device and we get nonsensical ``elapsed_time``
    # readings (or a hard error on cross-device record). Wrap event
    # creation, record, and synchronize in a single device guard.
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
    """Return bytes/sec throughput of CPU Adam on this host.

    Benchmarks ``deepspeed.ops.adam.DeepSpeedCPUAdam`` (the kernel the
    ``CpuFusedAdamAdapter`` uses in production) over a synthetic
    ``n_params``-long fp16 parameter + fp16 grad + fp32 optimizer state.
    Returns 0.0 if DeepSpeedCPUAdam cannot be imported or compiled —
    the cost model falls back to a hardcoded prior in that case.

    The default ``n_params = 10M`` yields ~200 MB of state (20 B/param) —
    well beyond L2/L3 cache sizes on any relevant host, so the measurement
    reflects sustained DRAM bandwidth rather than a cache-resident
    microbench.

    Parameters
    ----------
    n_params:
        Number of scalar fp16 parameters in the synthetic model.
    n_iters:
        Step invocations timed. The first is a warmup and is discarded
        from the median.

    Returns
    -------
    float
        Sustained Adam throughput in bytes/sec, where bytes = n_params *
        20 (see ``_ADAM_BYTES_PER_PARAM`` for the accounting breakdown).
        ``0.0`` on compile / import failure.
    """
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

    # DeepSpeedCPUAdam's ``__del__`` method calls
    # ``self.ds_opt_adam.destroy_adam(...)`` unconditionally; when the
    # constructor raises before ``ds_opt_adam`` is set (common on dev
    # rigs with CUDA toolchain mismatch), ``__del__`` raises
    # AttributeError on every GC pass. Python's unraisable-exception
    # handler fires, pytest's warning-capture hook intercepts it, and
    # the resulting traceback transitively pins autograd tensors from
    # the ProfilerTrace's traced forward pass (observed as +50 MB
    # ``memory_allocated`` on tiny-GPT2 in suite-level runs).
    # Neutralise the broken ``__del__`` before we try to instantiate so
    # any failed construction GC's cleanly.
    _orig_del = getattr(DeepSpeedCPUAdam, "__del__", None)

    def _safe_del(self: object) -> None:
        try:
            if hasattr(self, "ds_opt_adam"):
                _orig_del(self)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - suppress silently; dev-rig safety
            pass

    DeepSpeedCPUAdam.__del__ = _safe_del  # type: ignore[attr-defined]

    try:
        # Synthetic fp16 param + fp16 grad on CPU; DeepSpeedCPUAdam allocates
        # fp32 master + two fp32 momenta internally on first step.
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
            # Drop the exception traceback before returning so it can't pin
            # locals (and, via cycles, autograd tensors from the subsequent
            # traced forward pass — observed as a +50 MB ``memory_allocated``
            # ghost on tiny-GPT2 under pytest's unraisable-warning hook).
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
            # Re-populate grad each iter — Adam consumes it in-place but the
            # measurement should track the steady-state kernel cost.
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
        # Explicit cleanup — same rationale as measure_gpu_adam. We omit
        # gc.collect() here to avoid perturbing pytest's unraisable-exception
        # tracking of a failed DeepSpeedCPUAdam __del__ path.
        try:
            optim.zero_grad(set_to_none=True)
            optim.state.clear()
        except Exception:  # noqa: BLE001 - defensive
            pass
        del optim, param
        return float(bps)
    finally:
        # Restore the original ``__del__`` so that callers (and the rest of
        # the test session) see DeepSpeedCPUAdam's real finaliser instead of
        # our locally-patched ``_safe_del``. We unconditionally restore even
        # when the original was ``None`` (i.e. the class did not define a
        # ``__del__`` before we monkey-patched it) by deleting our override
        # so attribute lookup falls through to ``object.__del__``.
        if _orig_del is None:
            try:
                del DeepSpeedCPUAdam.__del__  # type: ignore[attr-defined]
            except AttributeError:
                pass
        else:
            DeepSpeedCPUAdam.__del__ = _orig_del  # type: ignore[attr-defined]


def measure_gpu_adam(
    device_idx: int = 0, n_params: int = 5_000_000, n_iters: int = 10
) -> float:
    """Return bytes/sec throughput of GPU Adam on this device.

    Uses the same fallback chain as
    :class:`axolotl.integrations.protrain.chunk.optim.GpuFusedAdamAdapter`:
    ``apex.optimizers.FusedAdam`` first (paper-cited), then
    ``torch.optim.AdamW`` (stock). Returns 0.0 only on a CUDA outage.

    Parameters
    ----------
    device_idx:
        CUDA ordinal.
    n_params:
        Scalar fp16 params in the synthetic model. 10M keeps state around
        200 MB — outside L2 on any 3090-class GPU, so the measurement
        reflects HBM bandwidth rather than L2 residency.
    n_iters:
        Timed step invocations. The first is a warmup, discarded.

    Returns
    -------
    float
        Throughput in bytes/sec (n_params * 20 / median_iter_s). 0.0 if
        no Adam implementation is constructible.
    """
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

    optim = None
    try:
        from apex.optimizers import FusedAdam  # type: ignore[import-not-found]

        optim = FusedAdam([param], lr=1e-4)
        backend = "apex.FusedAdam"
    except Exception:  # noqa: BLE001 - apex missing OR build mismatch
        pass

    if optim is None:
        try:
            # torch.optim.FusedAdam is a nightly-only alias; the stable
            # name is AdamW with fused=True on CUDA. Try that.
            optim = torch.optim.AdamW([param], lr=1e-4, fused=True)
            backend = "torch.optim.AdamW(fused=True)"
        except (TypeError, RuntimeError):
            # Older torch, or GPU without fused kernel support.
            optim = torch.optim.AdamW([param], lr=1e-4)
            backend = "torch.optim.AdamW"

    LOG.debug("measure_gpu_adam: backend=%s", backend)

    # Warmup + JIT.
    try:
        optim.step()
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.warning("measure_gpu_adam: warmup step failed (%s); returning 0.0", exc)
        return 0.0

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
    # caching allocator can reclaim the blocks; empty_cache is intentionally
    # NOT called because it forces the upcoming traced forward pass to
    # re-reserve memory from scratch, inflating its first-iter peak vs. the
    # ground-truth run that the reconstruct-peak test compares against.
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


# Payload sizes (bytes) swept by the multi-rank NCCL benchmark. Chosen to
# bracket the realistic ProTrain chunk sizes — S_chunk is selected from
# {32, 64, 128, 256} MiB per ``chunk/sizing.py``, so 64 MiB and 256 MiB sit
# at the centre of the sweep. The 1/4/16 MiB end captures the small-collective
# regime where launch latency dominates over bandwidth.
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
    """Measure NCCL gather + reduce latencies per payload size.

    Returns ``(gather_table, reduce_table)`` where each table maps payload
    bytes -> median collective time in seconds. Used by ``cost/runtime.py``
    to predict per-chunk all_gather / reduce_scatter cost for a given
    ``S_chunk`` choice.

    Single-rank fast path returns ``({}, {})`` — no NCCL traffic on
    ``world_size == 1`` and the searcher's communication term collapses.

    Multi-rank path requires the caller to have already initialized
    ``torch.distributed`` (any backend that supports the collectives below;
    NCCL is the only one ProTrain actually targets, but Gloo will also
    work for CPU-only smoke testing). Running under ``torchrun`` is the
    standard way; ``scripts/protrain/measure_nccl.py`` is a standalone
    driver that bootstraps a rendezvous on-demand.

    The benchmark uses ``all_gather_into_tensor`` (gather) and
    ``reduce_scatter_tensor`` (reduce) — these are the exact collectives
    ProTrain's M7 ZeRO-3 sharding path issues per chunk, so the measured
    times are directly applicable. ``n_warmup`` iterations bring the NCCL
    communicator + GPU IPC handles into steady state; the remaining
    ``n_iters`` are timed and the median is recorded.

    Parameters
    ----------
    world_size:
        Expected distributed world size. Sanity-checked against
        ``torch.distributed.get_world_size()`` to surface configuration
        bugs early (e.g. caller passed ``world_size=4`` but the rendezvous
        only sees 2 ranks).
    payload_sizes_bytes:
        Payload sizes to benchmark, in bytes. Default sweeps 1 MiB →
        256 MiB which brackets the typical S_chunk range.
    n_iters:
        Timed iterations per payload. Median is recorded.
    n_warmup:
        Warm-up iterations per payload (discarded).

    Returns
    -------
    tuple[dict[int, float], dict[int, float]]
        ``(gather_seconds_by_size, reduce_seconds_by_size)``.
    """
    if n_iters < 1:
        raise ValueError(f"measure_nccl: n_iters must be >= 1, got {n_iters}")
    if n_warmup < 0:
        raise ValueError(f"measure_nccl: n_warmup must be >= 0, got {n_warmup}")

    if world_size == 1:
        return ({}, {})

    import torch
    import torch.distributed as dist

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
    # Extract the integer ordinal so ``torch.cuda.device(device_idx)`` can
    # guard event construction + record + synchronize against a stale
    # ``current_device()`` under multi-GPU / CUDA_VISIBLE_DEVICES masking.
    device_idx = device.index if device.index is not None else 0

    gather_table: dict[int, float] = {}
    reduce_table: dict[int, float] = {}

    for payload_bytes in payload_sizes_bytes:
        # all_gather_into_tensor: each rank contributes one shard of size
        # payload/world_size, output is the full payload on every rank.
        # We size the SHARD to ``payload_bytes // world_size`` (rounded
        # DOWN to a multiple of ``element_size`` — both divisions are
        # integer floor) so the COMBINED output is at most payload_bytes.
        # ``world_size ∈ {2, 4, 8}`` for production use, all power-of-two,
        # so the rounding error is zero on the canonical payload grid;
        # the table is still keyed by the requested payload_bytes since
        # the cost model thinks in chunk-transfer units.
        element_size = 4  # float32
        elements_per_shard = max(1, (payload_bytes // world_size) // element_size)
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

        # Timed — wrap event construction + record + synchronize in one
        # device guard (cheaper than entering on each iter, equally correct).
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
        gather_table[payload_bytes] = statistics.median(gather_times)

        # reduce_scatter_tensor: input is full payload on every rank,
        # output is one shard per rank. Inverse of all_gather; same-shape
        # buffers reused.
        full_payload = torch.zeros(
            elements_per_shard * world_size,
            dtype=torch.float32,
            device=device,
        )
        reduced = torch.zeros(elements_per_shard, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(n_warmup):
            dist.reduce_scatter_tensor(reduced, full_payload)
        torch.cuda.synchronize(device)

        # Timed — wrap event construction + record + synchronize in one
        # device guard (cheaper than entering on each iter, equally correct).
        reduce_times: list[float] = []
        with torch.cuda.device(device_idx):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iters):
                start.record()
                dist.reduce_scatter_tensor(reduced, full_payload)
                end.record()
                torch.cuda.synchronize(device)
                reduce_times.append(start.elapsed_time(end) / 1000.0)
        reduce_table[payload_bytes] = statistics.median(reduce_times)

        del shard, gathered, full_payload, reduced
        # Free the four buffers' caching-allocator blocks before the next
        # payload bumps up. At world=4 / 256 MiB peak we hold ~640 MiB
        # live across the four; without empty_cache the allocator keeps
        # them reserved for a different stream's reuse, fragmenting the
        # pool for any future payload-grid expansion.
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
    """Return sustained fp16 compute throughput in TFLOPS for ``device_idx``.

    Runs a square fp16 matmul (``matrix_size`` × ``matrix_size``) over
    ``n_iters`` timed iterations and reports the median throughput in
    fp16-TFLOPS. The 3090 family lands around 45–55 TFLOPS sustained on
    a 4K GEMM (compared with the 71-TFLOPS peak rated number); a 3090 Ti
    is typically 5–10% faster on the same workload, which is exactly the
    spread the cost-model SKU calibration needs to absorb.

    Used by ``cost/runtime.py`` to scale per-op latencies when the cached
    trace was captured on a different SKU than the live training device:
    ``scale = trace.compute_rate_tflops / hw.gpu_compute_tflops``. Same-SKU
    runs see ``scale ≈ 1.0`` (the GEMM benchmark has ~2% noise floor) and
    the calibration is a no-op.

    Returns 0.0 on CUDA outage; the caller falls back to the trace's
    recorded value or the global default.

    Parameters
    ----------
    device_idx:
        CUDA device ordinal.
    matrix_size:
        Square matrix size for the synthetic GEMM. 4096 keeps a single
        matmul under ~270 MB (fp16 4096²) — well within any 3090's HBM
        and large enough that the kernel is firmly compute-bound.
    n_iters:
        Timed iterations. Median is reported.
    n_warmup:
        Warmup iterations (discarded). The first iter typically pays
        cuBLAS handle init + JIT cost.
    """
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

    # Warmup
    c = None
    for _ in range(n_warmup):
        c = a @ b
    torch.cuda.synchronize(device)
    if c is not None:
        del c

    # Timed — bind events + record + synchronize to ``device_idx`` so they
    # don't latch onto a stale ``current_device()`` under multi-GPU / masking.
    iter_s: list[float] = []
    with torch.cuda.device(device_idx):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(n_iters):
            start.record()
            c = a @ b
            end.record()
            torch.cuda.synchronize(device)
            iter_s.append(start.elapsed_time(end) / 1000.0)
    median_iter = statistics.median(iter_s)

    # FLOP count for a square matmul: 2 * N^3 (one multiply + one add per
    # element of the output, summed over the inner dim).
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
