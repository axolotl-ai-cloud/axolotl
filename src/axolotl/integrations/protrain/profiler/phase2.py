"""Phase-2 chunked-runtime profiler (paper §3.2 calibration loop).

The wrapper's first ``run_trace`` runs **without** the chunk manager
engaged — backward is skipped (``include_backward=False``) because on
7B+ models the unwrapped backward OOMs the 24 GiB card. The cost model
then falls back to a heuristic bwd/fwd ratio (1.0× LoRA, 2.0×
full-finetune) which on 7B-LoRA over-/under-shoots the actual chunked
backward by 25-30 %.

Phase-2 closes that gap. After the initial ``search()`` returns, the
wrapper builds the runtime under a conservative bootstrap config,
runs a short chunked steady-state ``forward → loss.backward() →
optim.step()`` measurement loop, and writes the median backward + step
overlap into ``ProfilerTrace.steady_bwd_chunked_wall_s`` and
``steady_step_overlap_s``. The cost model translates the measurement
across configs via ``phase2_n_checkpoint`` + ``phase2_per_block_recompute_s``
(D1b — see ``cost/runtime._bwd_compute_time_from_trace``).

The actual measurement loop lives here; the wrapper plumbing
(bootstrap → measure → splice → re-search → rebuild) lives in
``api/model_wrapper.py``.
"""

from __future__ import annotations

import copy
import statistics
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.types import (
    ChunkId,
    CostConfig,
    SearchResult,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

    from axolotl.integrations.protrain.types import (
        BlockStrategyMap,
        ChunkLayout,
        HardwareProfile,
        ProfilerTrace,
    )

LOG = get_logger(__name__)


# Number of warmup iterations discarded before timing starts. Three is
# enough to settle the buffer pool's LRU + gather/release cadence + CPU
# Adam's lazy state init, which all happen on the first forward/backward
# pass and would otherwise inflate the median.
_PHASE2_N_WARMUP = 3
# Number of timed iterations. Five gives a stable median on the 7B-LoRA
# canonical workload (per-iter variance ~5%); larger N adds latency
# without visibly tightening the median.
_PHASE2_N_ITERS = 5


def _min_n_buffer_for_layout(layout: "ChunkLayout", n_persist: int) -> int:
    """Minimum pool size needed for adjacent-block prefetch at ``n_persist``."""
    if n_persist >= layout.N_chunk:
        return 0
    persistent: set[ChunkId] = {ChunkId(i) for i in range(n_persist)}
    block_ids = sorted(layout.block_to_chunks.keys())
    if not block_ids:
        return 0
    need = 0
    for i, bid in enumerate(block_ids):
        cur_np = [c for c in layout.block_to_chunks.get(bid, ()) if c not in persistent]
        nxt_np: list[ChunkId] = []
        if i + 1 < len(block_ids):
            nxt_np = [
                c
                for c in layout.block_to_chunks.get(block_ids[i + 1], ())
                if c not in persistent
            ]
        need = max(need, len({*cur_np, *nxt_np}))
    return max(1, need)


def select_bootstrap_config(
    *,
    initial_result: SearchResult,
    layout: "ChunkLayout",
    n_block: int,
    capacity_bytes: int,
    trace: "ProfilerTrace",
    hw: "HardwareProfile",
) -> tuple[CostConfig, "BlockStrategyMap"]:
    """Pick a conservative bootstrap config that's guaranteed to fit.

    Spec: ``n_persist=0``, ``n_swap=0``, ``n_checkpoint=N_block``,
    ``n_buffer=min(layout.N_chunk, max(initial_result.cfg.n_buffer,
    _min_n_buffer_for_layout(layout, n_persist=0)))``. This biases hard
    toward memory savings (zero persistence, full activation
    checkpointing) while keeping ``n_buffer`` large enough to satisfy
    the layout's adjacent-block prefetch requirement and never
    exceeding the total chunk count.

    Lowering ``n_persist`` to zero (vs. carrying over the searcher's
    higher-persistence pick) is what makes this a calibration baseline
    for low-persistence offload configs — the phase-2 measurement is
    later reused to correct the cost model's replay-time chunk-gather
    estimate, which would be under-counted if we measured at high
    persistence. ``n_buffer`` is floored at the searcher's pick so we
    don't regress the prefetch window.

    Validates the candidate against ``estimate_peak``; if the peak
    exceeds capacity, fall back to the search's own first pick (which
    by construction passed the capacity gate). This second-line
    defense covers degenerate models where even max-CKPT + zero-
    persistent doesn't fit — those would already have crashed before
    phase-2, but be defensive.
    """
    from axolotl.integrations.protrain.block.layout_rules import assign_modes
    from axolotl.integrations.protrain.cost.memory import estimate_peak

    # Measure a conservative low-persistence, all-CKPT runtime. The
    # phase-2 measurement is later used as a calibration baseline for
    # low-persistence offload configs, so using the initial search's
    # high-persistence pick can under-count replay-time chunk gathers by
    # several multiples. Keep the searcher's n_buffer as a lower bound,
    # then raise it if lowering n_persist increases the adjacent-block
    # prefetch window.
    min_buffer = _min_n_buffer_for_layout(layout, 0)
    bootstrap_cfg = CostConfig(
        n_persist=0,
        n_buffer=min(
            layout.N_chunk,
            max(initial_result.cfg.n_buffer, min_buffer),
        ),
        n_swap=0,
        n_checkpoint=n_block,
    )
    bootstrap_block_map = assign_modes(0, n_block, n_block)

    candidate_peak = estimate_peak(
        bootstrap_cfg, trace, layout, bootstrap_block_map, hw
    )
    if candidate_peak <= capacity_bytes:
        LOG.info(
            "Phase-2 bootstrap config: n_persist=%d n_buffer=%d "
            "n_checkpoint=%d (peak %.2f GB <= capacity %.2f GB)",
            bootstrap_cfg.n_persist,
            bootstrap_cfg.n_buffer,
            bootstrap_cfg.n_checkpoint,
            candidate_peak / (1 << 30),
            capacity_bytes / (1 << 30),
        )
        return bootstrap_cfg, bootstrap_block_map

    LOG.warning(
        "Phase-2 bootstrap formula (n_persist=%d n_buffer=%d "
        "n_checkpoint=%d) predicts peak %.2f GB > capacity %.2f GB; "
        "falling back to the searcher's first pick which passed the "
        "capacity gate by construction.",
        bootstrap_cfg.n_persist,
        bootstrap_cfg.n_buffer,
        bootstrap_cfg.n_checkpoint,
        candidate_peak / (1 << 30),
        capacity_bytes / (1 << 30),
    )
    return initial_result.cfg, initial_result.block_map


def _clone_state_dict(state):
    """Recursively clone every tensor in a (possibly nested) state_dict.

    ``Module.state_dict()`` and ``Optimizer.state_dict()`` both return
    *aliased references* to the live parameter / optimizer tensors —
    iterating them and calling ``optimizer.step()`` mutates those
    tensors in-place, so a bare snapshot is silently mutated by the
    timed loop and ``load_state_dict()`` would restore from already-
    advanced state. We walk the structure and ``.detach().clone()``
    each tensor so the snapshot has independent storage; non-tensor
    leaves (ints, floats, ``ParamGroup`` configs, etc.) are
    ``copy.deepcopy``'d so dicts/lists also get independent identity.

    Recurses through ``dict``/``list``/``tuple`` containers because
    ``Optimizer.state_dict()`` is shaped
    ``{"state": {param_id: {tensor_key: tensor, ...}}, "param_groups": [...]}``.
    """
    import torch

    if torch.is_tensor(state):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: _clone_state_dict(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        cloned = [_clone_state_dict(v) for v in state]
        return type(state)(cloned) if isinstance(state, tuple) else cloned
    return copy.deepcopy(state)


def measure_chunked_steady(
    *,
    model: "nn.Module",
    batch: dict,
    optimizer: "torch.optim.Optimizer",
    n_warmup: int = _PHASE2_N_WARMUP,
    n_iters: int = _PHASE2_N_ITERS,
) -> tuple[float, float, float, int]:
    """Run a chunked steady-state ``fwd → bwd → step`` loop and time it.

    Times the forward, backward, and post-backward optimizer step using
    ``torch.cuda.Event`` pairs (same convention as
    :mod:`profiler.hw_bench` for ``measure_compute_rate`` /
    ``measure_cpu_adam`` / ``measure_gpu_adam``). The optimizer step
    timing window includes the wait for the asynchronous CPU FusedAdam
    that the per-param grad hooks kick off during backward — so it
    captures the bwd↔step overlap envelope, not the cumulative compute.

    The forward window measures the full chunked-runtime forward
    (compute + chunk-prefetch / gather overhead inherent to the chunk
    manager). Closes the residual forward over-prediction left over
    after the v10 backward calibration.

    Returns
    -------
    (steady_fwd_chunked_wall_s, steady_bwd_chunked_wall_s,
    steady_step_overlap_s, steady_phase2_peak_bytes)
        Median across ``n_iters`` timed iterations. ``n_warmup``
        iterations are discarded — they pay one-time costs (chunk
        manager LRU settling, CPU Adam state lazy init, autograd
        graph construction) that would inflate the median. Peak bytes
        are the CUDA high-water mark across the timed loop.
    """
    import torch

    if n_warmup < 0 or n_iters <= 0:
        raise ValueError("n_warmup must be >= 0 and n_iters must be > 0")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase-2 measurement requires CUDA; got torch.cuda.is_available() == False"
        )

    model.train()
    # Bind every CUDA timing/memory API call to the model's device so a
    # future refactor that changes the current-device context between
    # plugin setup and measurement cannot silently measure the wrong GPU.
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Phase-2 measurement expected a CUDA model, got {device!r}")

    with torch.cuda.device(device):
        # Snapshot model + optimizer state BEFORE warmup so the
        # measurement (which calls ``optimizer.step()`` and mutates
        # parameters) is non-destructive: training resumes from the
        # same initial state after the profiler returns. The snapshot
        # itself is excluded from the timed region — captured before
        # warmup, restored after the timed loop.
        #
        # ``state_dict()`` returns *aliased* tensor references — the
        # subsequent ``optimizer.step()`` calls would mutate those
        # tensors in-place and silently advance the snapshot, so we
        # deep-clone every tensor (independent storage) before warmup.
        # Synchronize first so any in-flight kernels finish writing
        # before we read parameters / optimizer state into the clone.
        torch.cuda.synchronize(device)
        model_state = _clone_state_dict(model.state_dict())
        optim_state = _clone_state_dict(optimizer.state_dict())
        # Start from a clean grad state so leftover grads from prior
        # trace work (e.g. the phase-1 profile pass) cannot pollute
        # the first warmup step's peak-memory and timing samples.
        optimizer.zero_grad(set_to_none=True)
        # Wrap warmup + timed loop in try/finally so an exception
        # mid-measurement (OOM, NaN loss, kernel error) still rolls
        # the model + optimizer back to the pre-measurement state.
        # Without this the caller would inherit a partially-advanced
        # optimizer + parameters that bear no relation to the
        # checkpoint they handed in.
        try:
            # Warmup — discard timings.
            for _ in range(n_warmup):
                out = model(**batch)
                loss = _extract_loss(out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            # Re-zero after the peak-stats reset: warmup left grads at
            # ``None`` already, but be explicit so the timed loop's
            # first iteration always starts from the same grad state
            # regardless of ``n_warmup``.
            optimizer.zero_grad(set_to_none=True)

            fwd_times_s: list[float] = []
            bwd_times_s: list[float] = []
            step_times_s: list[float] = []
            for _ in range(n_iters):
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)
                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)

                fwd_start.record()
                out = model(**batch)
                loss = _extract_loss(out)
                fwd_end.record()

                bwd_start.record()
                loss.backward()
                bwd_end.record()
                optimizer.step()
                step_end.record()

                torch.cuda.synchronize(device)
                fwd_times_s.append(fwd_start.elapsed_time(fwd_end) / 1000.0)
                bwd_times_s.append(bwd_start.elapsed_time(bwd_end) / 1000.0)
                step_times_s.append(bwd_end.elapsed_time(step_end) / 1000.0)

                optimizer.zero_grad(set_to_none=True)

            fwd_median = statistics.median(fwd_times_s)
            bwd_median = statistics.median(bwd_times_s)
            step_median = statistics.median(step_times_s)
            peak_bytes = int(torch.cuda.max_memory_allocated(device))
        finally:
            # Restore the pre-measurement model + optimizer state so
            # the profiler is non-destructive: ``optimizer.step()``
            # calls in warmup + timed loops mutated parameters and
            # optimizer state. Synchronize first so any in-flight
            # kernels referencing these tensors complete before we
            # overwrite them, and again after so the load is visible
            # to the caller before we return. ``load_state_dict``
            # copies values into the live tensors, so as long as the
            # snapshot has independent storage (it does — see
            # ``_clone_state_dict``) the rollback is exact.
            torch.cuda.synchronize(device)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
    LOG.info(
        "Phase-2 chunked-runtime measurement: "
        "steady_fwd_chunked_wall_s=%.4f (n=%d, samples=%s) "
        "steady_bwd_chunked_wall_s=%.4f (samples=%s) "
        "steady_step_overlap_s=%.4f (samples=%s) "
        "steady_phase2_peak_bytes=%.2f GB",
        fwd_median,
        n_iters,
        ["%.4f" % t for t in fwd_times_s],
        bwd_median,
        ["%.4f" % t for t in bwd_times_s],
        step_median,
        ["%.4f" % t for t in step_times_s],
        peak_bytes / (1 << 30),
    )
    return fwd_median, bwd_median, step_median, peak_bytes


def estimate_per_block_recompute_s(trace: "ProfilerTrace", n_block: int) -> float:
    """Mean per-block forward compute time (≡ recompute under CKPT).

    Uses :func:`cost.runtime._fwd_compute_time_from_trace` to derive
    per-block forward time from the trace's measured op latencies (or
    the activation-size roofline proxy when latencies are absent).
    Returns the mean across blocks — phase-2's translation formula
    works in mean-per-block units because the cost model approximates
    per-block recompute as a uniform per-block term.

    Returns 0.0 when ``n_block == 0`` or when the trace has no op
    latencies AND no activation sizes (degenerate trace — would only
    happen in a unit test fixture, never on a live profile).
    """
    from axolotl.integrations.protrain.cost.runtime import (
        _fwd_compute_time_from_trace,
    )

    if n_block <= 0:
        return 0.0
    t_fwd_total, per_block_compute, _used_measured = _fwd_compute_time_from_trace(trace)
    if per_block_compute:
        # Mean of measured per-block times — this is what the cost
        # model adds per CKPT block via ``per_block_compute.get(bid)``.
        return sum(per_block_compute.values()) / max(1, len(per_block_compute))
    if t_fwd_total > 0.0:
        # Fallback: divide aggregate forward by N_block. Less accurate
        # but the cost model uses the same fallback (activation-size
        # roofline) per block — we maintain symmetry.
        return t_fwd_total / n_block
    return 0.0


def _extract_loss(out) -> "torch.Tensor":
    """Pull a backwards-able scalar loss out of a HuggingFace forward output.

    Delegates to the shared ``trace._extract_loss`` so the supported
    output shapes stay in sync: HF attribute-style (``CausalLMOutput.loss``),
    dict-style (``out["loss"]``), raw scalar/non-scalar ``torch.Tensor``,
    and tuple/list whose first scalar tensor is the loss. Raises
    ``TypeError`` (from the shared helper) if none of those match —
    phase-2 needs a ``.backward()``-able tensor.
    """
    # Local import keeps phase2 importable without forcing trace at module
    # load time; trace.py does not import phase2 so there's no cycle.
    from axolotl.integrations.protrain.profiler.trace import (
        _extract_loss as _trace_extract_loss,
    )

    return _trace_extract_loss(out)


__all__ = [
    "measure_chunked_steady",
    "select_bootstrap_config",
    "estimate_per_block_recompute_s",
]
