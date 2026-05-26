"""Runtime (wall-clock) cost estimator for the ProTrain searcher."""

from __future__ import annotations

import logging

from axolotl.integrations.protrain.cost.bandwidth import (
    effective_bw,
    effective_bw_for_chunk,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Fallback compute throughput proxy used only when op_latencies are missing.
_COMPUTE_BYTES_PER_SEC: float = 3.0e11  # ~300 GB/s, rough 3090 effective

# CPU-Adam throughput fallback when hw_bench measurement returned 0.0.
_CPU_ADAM_FALLBACK: float = 8.0e9

# GPU FusedAdam throughput fallback; ~3090 HBM sustained.
_GPU_ADAM_FALLBACK: float = 5.0e11

# One-shot warning gates for the searcher hot-loop.
_WARNED_MODEL_STATE_MISSING: bool = False
_WARNED_CPU_ADAM_UNAVAILABLE: bool = False
_WARNED_GPU_ADAM_FALLBACK: bool = False
_WARNED_SKU_SCALE_CLAMPED: bool = False
_WARNED_HOOK_SCALE_CLAMPED: bool = False
_WARNED_APPROXIMATE_COMPUTE_PROXY: bool = False

# Backward-vs-forward compute ratio when the trace has no per-block backward split.
_BWD_FWD_COMPUTE_RATIO: float = 2.0

# Hook-less/hooked forward wall-time scale clamp; <0.3 over-corrects, >1.0 indicates measurement glitch.
_HOOK_SCALE_MIN: float = 0.3
_HOOK_SCALE_MAX: float = 1.0

# SKU compute-rate scale clamp; cross-3090 variation is ~5-10%.
_SKU_SCALE_MIN: float = 0.5
_SKU_SCALE_MAX: float = 2.0

# Infeasibility sentinel — estimate_runtime propagates as inf so searcher rejects.
_INF_COMPONENTS: tuple[float, float, float, float, bool, bool] = (
    float("inf"),
    float("inf"),
    float("inf"),
    float("inf"),
    False,
    False,
)


def _sku_compute_scale(trace: ProfilerTrace, hw: HardwareProfile) -> float:
    """Return the trace-vs-live compute-rate ratio, clamped."""
    if trace.compute_rate_tflops <= 0.0 or hw.gpu_compute_tflops <= 0.0:
        return 1.0
    raw = trace.compute_rate_tflops / hw.gpu_compute_tflops
    if raw < _SKU_SCALE_MIN or raw > _SKU_SCALE_MAX:
        global _WARNED_SKU_SCALE_CLAMPED
        if not _WARNED_SKU_SCALE_CLAMPED:
            LOG.warning(
                "SKU compute-rate scale out of sane range (%.3f = trace %.1f / "
                "live %.1f TFLOPS); clamping to [%.2f, %.2f]. Treat with "
                "suspicion — likely a measurement glitch on one of the two SKUs. "
                "(further occurrences suppressed)",
                raw,
                trace.compute_rate_tflops,
                hw.gpu_compute_tflops,
                _SKU_SCALE_MIN,
                _SKU_SCALE_MAX,
            )
            _WARNED_SKU_SCALE_CLAMPED = True
    return max(_SKU_SCALE_MIN, min(_SKU_SCALE_MAX, raw))


def _hook_scale_factor(trace: ProfilerTrace) -> float:
    """Return the steady/hooked forward wall-time ratio, clamped to a sane range."""
    if trace.hooked_fwd_wall_s <= 0.0 or trace.steady_fwd_wall_s <= 0.0:
        return 1.0
    raw = trace.steady_fwd_wall_s / trace.hooked_fwd_wall_s
    if raw > _HOOK_SCALE_MAX or raw < _HOOK_SCALE_MIN:
        global _WARNED_HOOK_SCALE_CLAMPED
        if not _WARNED_HOOK_SCALE_CLAMPED:
            LOG.warning(
                "hook-scale ratio out of sane range (%.3f = steady %.4fs / hooked "
                "%.4fs); clamping to [%.2f, %.2f] (further occurrences suppressed)",
                raw,
                trace.steady_fwd_wall_s,
                trace.hooked_fwd_wall_s,
                _HOOK_SCALE_MIN,
                _HOOK_SCALE_MAX,
            )
            _WARNED_HOOK_SCALE_CLAMPED = True
    return max(_HOOK_SCALE_MIN, min(_HOOK_SCALE_MAX, raw))


def _compute_time(activation_bytes: int) -> float:
    """Rough compute time proxy — fallback when op_latencies missing."""
    return activation_bytes / _COMPUTE_BYTES_PER_SEC


def _block_compute_time(trace: ProfilerTrace, block_id: BlockId) -> float:
    """Wall-clock forward compute for one block; sum of measured op latencies."""
    total_s = 0.0
    for op in trace.op_order:
        if op.block_id != block_id or not op.is_forward:
            continue
        total_s += trace.op_latencies.get(op.op_id, 0.0)
    return total_s


def _fwd_compute_time_from_trace(
    trace: ProfilerTrace,
    cfg: CostConfig | None = None,
) -> tuple[float, dict[BlockId, float], bool, float]:
    """Return (total_fwd_compute_s, per_block_compute_s, used_measured, fwd_compute_base_s)."""
    per_block: dict[BlockId, float] = {}
    total = 0.0
    # Always compute the roofline reference; cheap, and used as a sanity cap.
    roofline_per_block: dict[BlockId, float] = {}
    roofline_total = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        t = _compute_time(act_sz)
        roofline_per_block[bid] = t
        roofline_total += t

    if trace.op_latencies:
        hooked_per_block: dict[BlockId, float] = {}
        hooked_total = 0.0
        for op in trace.op_order:
            if not op.is_forward or op.block_id is None:
                continue
            lat = trace.op_latencies.get(op.op_id)
            if lat is None:
                continue
            hooked_per_block[op.block_id] = hooked_per_block.get(op.block_id, 0.0) + lat
            hooked_total += lat
        for bid_raw in trace.activation_sizes:
            bid = BlockId(int(bid_raw))
            hooked_per_block.setdefault(bid, 0.0)

        # Apply clamped hook-dispatch scale; legacy traces get identity.
        scale = _hook_scale_factor(trace)
        per_block = {bid: v * scale for bid, v in hooked_per_block.items()}
        total = hooked_total * scale

        if total > 0.0:
            # Cap at steady_fwd_wall_s (or 2x roofline fallback) to prevent runaway predictions.
            cap = 0.0
            if trace.steady_fwd_wall_s > 0.0:
                cap = trace.steady_fwd_wall_s
            elif roofline_total > 0.0:
                cap = 2.0 * roofline_total
            if cap > 0.0 and total > cap:
                safety = cap / total
                per_block = {bid: v * safety for bid, v in per_block.items()}
                total = cap
            # Phase-2 chunked-wall override (n_swap==0 gate); preserve pre-override base for bwd fallback.
            fwd_compute_base = total
            if trace.steady_fwd_chunked_wall_s > 0.0 and (
                cfg is None or cfg.n_swap == 0
            ):
                total = trace.steady_fwd_chunked_wall_s
            return total, per_block, True, fwd_compute_base

    # Pure roofline fallback; no measurements available.
    return roofline_total, roofline_per_block, False, roofline_total


def _bwd_compute_time_from_trace(
    trace: ProfilerTrace,
    t_fwd_total: float,
    cfg: CostConfig | None = None,
) -> float:
    """Return the aggregate backward compute time in seconds."""
    # Phase-2 chunked measurement gated on cfg.n_swap == 0.
    if (
        trace.steady_bwd_chunked_wall_s > 0.0
        and (trace.phase2_n_checkpoint == 0 or trace.phase2_per_block_recompute_s > 0.0)
        and (cfg is None or cfg.n_swap == 0)
    ):
        bootstrap_recompute = (
            trace.phase2_n_checkpoint * trace.phase2_per_block_recompute_s
        )
        base = max(0.0, trace.steady_bwd_chunked_wall_s - bootstrap_recompute)
        return base
    # Steady unwrapped measurement.
    if trace.steady_bwd_wall_s > 0.0 and trace.steady_fwd_wall_s > 0.0:
        measured_ratio = trace.steady_bwd_wall_s / trace.steady_fwd_wall_s
        # Clamp to [1.0, 3.0]: LoRA hits 1.0x, full-FT with attn recomp ~3x.
        measured_ratio = max(1.0, min(3.0, measured_ratio))
        return t_fwd_total * measured_ratio
    # Trainable-fraction heuristic.
    if 0.0 < trace.trainable_param_fraction < 0.05:
        return t_fwd_total * 1.0
    return t_fwd_total * _BWD_FWD_COMPUTE_RATIO


def _comm_time_chunk(
    S_chunk: int,
    eff_h2d: float,
    eff_d2h: float,
    nccl_gather_s: float,
    *,
    is_backward: bool,
    buffer_cached: bool,
    nccl_reduce_s: float = 0.0,
) -> float:
    """Return the communication time for a single non-persistent chunk."""
    collective = nccl_gather_s

    # Fail closed: zero PCIe bandwidth would make a broken candidate look cheaper than a valid one.
    if not is_backward:
        if eff_h2d <= 0:
            return float("inf")
        return collective + S_chunk / eff_h2d

    if eff_d2h <= 0 or (not buffer_cached and eff_h2d <= 0):
        return float("inf")

    d2h = S_chunk / eff_d2h
    if buffer_cached:
        # Cache-hit: skip gather + H2D; reduce_scatter still required (Eq. 6).
        return d2h + nccl_reduce_s
    return collective + S_chunk / eff_h2d + d2h + nccl_reduce_s


def _pick_nccl(nccl_table: dict, payload_bytes: int) -> float:
    """Look up the nearest payload size in an NCCL latency table."""
    if not nccl_table:
        return 0.0
    best = min(nccl_table.keys(), key=lambda k: abs(int(k) - payload_bytes))
    return float(nccl_table[best])


def estimate_runtime(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,
    *,
    chunk_bw_table: (
        dict[int, tuple[tuple[float, float], tuple[float, float]]] | None
    ) = None,
) -> float:
    """Estimate wall-clock iteration time in seconds.

    ``chunk_bw_table`` (optional): precomputed per-chunk effective bandwidths
    ``{chunk_id: ((fwd_h2d, fwd_d2h), (bwd_h2d, bwd_d2h))}``. Equivalent to
    calling ``effective_bw_for_chunk`` per chunk, hoisted out of the searcher
    inner loop where ``(n_swap, block_map, layout, hw)`` are loop-invariant.
    """
    t_fwd, t_bwd, t_gpu_optim, t_cpu_optim, fwd_used_phase2, bwd_used_phase2 = (
        _estimate_runtime_components(
            cfg, trace, layout, block_map, hw, chunk_bw_table=chunk_bw_table
        )
    )
    if t_fwd == float("inf") or t_bwd == float("inf"):
        return float("inf")
    if t_cpu_optim == float("inf"):
        return float("inf")
    return _compose_t_iter_with_alpha_calibration(
        cfg=cfg,
        trace=trace,
        t_fwd=t_fwd,
        t_bwd=t_bwd,
        t_gpu_optim=t_gpu_optim,
        t_cpu_optim=t_cpu_optim,
        fwd_used_phase2_override=fwd_used_phase2,
        bwd_used_phase2_override=bwd_used_phase2,
    )


# Per-component alpha clamp bounds; [0.5, 2.0] window catches noise outside [0.3, 3.0].
_PHASE2_ALPHA_CLAMP_MIN: float = 0.5
_PHASE2_ALPHA_CLAMP_MAX: float = 2.0
_PHASE2_ALPHA_NOISE_FLOOR: float = 0.3
_PHASE2_ALPHA_NOISE_CEILING: float = 3.0
_WARNED_PHASE2_ALPHA_NOISY: bool = False

# Residual-alpha clamp bounds; analytical under-counts overhead so inflate side wider.
_PHASE2_RESIDUAL_CLAMP_MIN: float = 0.8
_PHASE2_RESIDUAL_CLAMP_MAX: float = 2.0
_PHASE2_RESIDUAL_NOISE_FLOOR: float = 0.5
_PHASE2_RESIDUAL_NOISE_CEILING: float = 5.0
_WARNED_PHASE2_RESIDUAL_NOISY: bool = False


def _clamp_alpha(alpha: float, name: str) -> float:
    """Clamp a per-component alpha to [0.5, 2.0]; warn once if outside [0.3, 3.0]."""
    if alpha < _PHASE2_ALPHA_NOISE_FLOOR or alpha > _PHASE2_ALPHA_NOISE_CEILING:
        global _WARNED_PHASE2_ALPHA_NOISY
        if not _WARNED_PHASE2_ALPHA_NOISY:
            LOG.warning(
                "estimate_runtime: phase-2 per-component %s = %.3f is "
                "outside the noise envelope [%.2f, %.2f] — measurement noise "
                "may be dominating signal. Clamping to [%.2f, %.2f]; "
                "investigate phase-2 measurement variance if predictions "
                "regress.",
                name,
                alpha,
                _PHASE2_ALPHA_NOISE_FLOOR,
                _PHASE2_ALPHA_NOISE_CEILING,
                _PHASE2_ALPHA_CLAMP_MIN,
                _PHASE2_ALPHA_CLAMP_MAX,
            )
            _WARNED_PHASE2_ALPHA_NOISY = True
    return max(_PHASE2_ALPHA_CLAMP_MIN, min(_PHASE2_ALPHA_CLAMP_MAX, alpha))


# ±1 n_checkpoint tolerance; phase-2 may re-eval recompute cost.
_STRUCTURE_MATCH_NCKPT_TOL: int = 1


def _structure_match(
    cfg: "CostConfig", trace: "ProfilerTrace", n_ckpt_prod: int
) -> bool:
    """Whether prod cfg matches boot's shape closely enough for alpha deflation to transfer."""
    boot_n_persist = int(getattr(trace, "phase2_n_persist", -1))
    boot_n_checkpoint = int(getattr(trace, "phase2_n_checkpoint", -1))
    if boot_n_persist < 0 or boot_n_checkpoint < 0:
        # No boot cfg signal: fail-safe to inflate-only.
        return False
    return (
        int(cfg.n_persist) == boot_n_persist
        and int(cfg.n_swap) == 0  # boot is always n_swap=0
        and abs(int(n_ckpt_prod) - boot_n_checkpoint) <= _STRUCTURE_MATCH_NCKPT_TOL
    )


def _clamp_alpha_inflate_only(alpha: float, name: str) -> float:
    """Clamp raw alpha to [1.0, 2.0] - inflate-only - when shape gate fires."""
    if alpha < _PHASE2_ALPHA_NOISE_FLOOR or alpha > _PHASE2_ALPHA_NOISE_CEILING:
        global _WARNED_PHASE2_ALPHA_NOISY
        if not _WARNED_PHASE2_ALPHA_NOISY:
            LOG.warning(
                "estimate_runtime: phase-2 per-component %s = %.3f is "
                "outside the noise envelope [%.2f, %.2f] — measurement noise "
                "may be dominating signal. Clamping to [1.00, %.2f] "
                "(shape-gate active); investigate phase-2 measurement "
                "variance if predictions regress.",
                name,
                alpha,
                _PHASE2_ALPHA_NOISE_FLOOR,
                _PHASE2_ALPHA_NOISE_CEILING,
                _PHASE2_ALPHA_CLAMP_MAX,
            )
            _WARNED_PHASE2_ALPHA_NOISY = True
    return max(1.0, min(_PHASE2_ALPHA_CLAMP_MAX, alpha))


def _clamp_residual_alpha(alpha: float) -> float:
    """Clamp the residual whole-iter alpha to [0.8, 2.0]; warn once if outside [0.5, 5.0]."""
    if alpha < _PHASE2_RESIDUAL_NOISE_FLOOR or alpha > _PHASE2_RESIDUAL_NOISE_CEILING:
        global _WARNED_PHASE2_RESIDUAL_NOISY
        if not _WARNED_PHASE2_RESIDUAL_NOISY:
            LOG.warning(
                "estimate_runtime: phase-2 alpha_residual = %.3f is outside the "
                "noise envelope [%.2f, %.2f] — either phase-2 iter measurement "
                "is noisy or a per-component term is missing. Clamping to "
                "[%.2f, %.2f]; investigate phase-2 measurement quality if "
                "predictions regress.",
                alpha,
                _PHASE2_RESIDUAL_NOISE_FLOOR,
                _PHASE2_RESIDUAL_NOISE_CEILING,
                _PHASE2_RESIDUAL_CLAMP_MIN,
                _PHASE2_RESIDUAL_CLAMP_MAX,
            )
            _WARNED_PHASE2_RESIDUAL_NOISY = True
    return max(_PHASE2_RESIDUAL_CLAMP_MIN, min(_PHASE2_RESIDUAL_CLAMP_MAX, alpha))


def _compose_t_iter_with_alpha_calibration(
    *,
    cfg: CostConfig,
    trace: ProfilerTrace,
    t_fwd: float,
    t_bwd: float,
    t_gpu_optim: float,
    t_cpu_optim: float,
    fwd_used_phase2_override: bool,
    bwd_used_phase2_override: bool,
) -> float:
    """Compose t_iter from per-component times, applying phase-2 alpha calibration."""
    has_per_component = (
        getattr(trace, "phase2_analytical_fwd_s", 0.0) > 0.0
        and getattr(trace, "phase2_analytical_bwd_s", 0.0) > 0.0
        and getattr(trace, "phase2_analytical_step_s", 0.0) > 0.0
        and getattr(trace, "phase2_fwd_s", 0.0) > 0.0
        and getattr(trace, "phase2_bwd_s", 0.0) > 0.0
        and getattr(trace, "phase2_step_s", 0.0) > 0.0
    )
    if has_per_component:
        a_fwd_raw = trace.phase2_fwd_s / trace.phase2_analytical_fwd_s
        a_bwd_raw = trace.phase2_bwd_s / trace.phase2_analytical_bwd_s
        a_opt_raw = trace.phase2_step_s / trace.phase2_analytical_step_s
        # Inflate-only when prod shape differs from boot; boot biases don't transfer.
        shape_matches = _structure_match(cfg, trace, int(cfg.n_checkpoint))
        if shape_matches:
            a_fwd = _clamp_alpha(a_fwd_raw, "alpha_fwd")
            a_bwd = _clamp_alpha(a_bwd_raw, "alpha_bwd")
            a_opt = _clamp_alpha(a_opt_raw, "alpha_opt")
        else:
            a_fwd = _clamp_alpha_inflate_only(a_fwd_raw, "alpha_fwd")
            a_bwd = _clamp_alpha_inflate_only(a_bwd_raw, "alpha_bwd")
            a_opt = _clamp_alpha_inflate_only(a_opt_raw, "alpha_opt")
        # Skip alpha on the override path - measurement-anchored already.
        a_fwd_eff = 1.0 if fwd_used_phase2_override else a_fwd
        a_bwd_eff = 1.0 if bwd_used_phase2_override else a_bwd
        t_fwd_cal = a_fwd_eff * t_fwd
        t_bwd_cal = a_bwd_eff * t_bwd
        t_gpu_cal = a_opt * t_gpu_optim
        t_cpu_cal = a_opt * t_cpu_optim
        t_step_cal = max(t_gpu_cal, t_cpu_cal)
        t_iter = t_fwd_cal + t_bwd_cal + t_step_cal

        # Residual alpha absorbs whole-iter overhead the per-component baselines don't model.
        anchor = float(getattr(trace, "phase2_per_comp_pred_iter_s", 0.0))
        measured = float(getattr(trace, "phase2_iter_s", 0.0))
        if anchor > 0.0 and measured > 0.0 and shape_matches:
            # Shape-match gate's symmetric companion: residual alpha suppressed on shape change.
            a_residual = _clamp_residual_alpha(measured / max(anchor, 1e-9))
        else:
            a_residual = 1.0
        t_iter_pre_residual = t_iter
        t_iter = a_residual * t_iter
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "estimate_runtime: phase-2 per-component alpha applied "
                "(shape_matches=%s, alpha_fwd=%.3f alpha_bwd=%.3f alpha_opt=%.3f, "
                "alpha_residual=%.3f, fwd_override=%s bwd_override=%s, "
                "t_fwd=%.4fs t_bwd=%.4fs t_gpu=%.4fs t_cpu=%.4fs "
                "t_step=%.4fs "
                "-> t_iter_pre_residual=%.4fs -> t_iter=%.4fs)",
                shape_matches,
                a_fwd,
                a_bwd,
                a_opt,
                a_residual,
                fwd_used_phase2_override,
                bwd_used_phase2_override,
                t_fwd_cal,
                t_bwd_cal,
                t_gpu_cal,
                t_cpu_cal,
                t_step_cal,
                t_iter_pre_residual,
                t_iter,
            )
        return t_iter
    # Single-alpha legacy fallback for in-memory traces without per-component fields.
    t_iter = t_fwd + t_bwd + max(t_gpu_optim, t_cpu_optim)
    used_analytical_path = (not fwd_used_phase2_override) or (
        not bwd_used_phase2_override
    )
    if (
        used_analytical_path
        and getattr(trace, "phase2_iter_s", 0.0) > 0.0
        and getattr(trace, "phase2_analytical_iter_s", 0.0) > 0.0
    ):
        alpha = _clamp_alpha(
            trace.phase2_iter_s / trace.phase2_analytical_iter_s,
            "alpha_legacy_single",
        )
        t_iter_pre = t_iter
        t_iter = t_iter * alpha
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "estimate_runtime: phase-2 alpha_legacy_single applied "
                "(alpha_legacy_single=%.3f, %.4fs -> %.4fs)",
                alpha,
                t_iter_pre,
                t_iter,
            )
    return t_iter


def _estimate_runtime_components(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,
    *,
    chunk_bw_table: (
        dict[int, tuple[tuple[float, float], tuple[float, float]]] | None
    ) = None,
) -> tuple[float, float, float, float, bool, bool]:
    """Compute the four runtime components (no calibration applied).

    ``chunk_bw_table`` short-circuits the per-chunk ``effective_bw_for_chunk``
    calls when the caller has already hoisted them out of an outer loop.
    """
    full_h2d = hw.pcie_h2d_bps

    # Worst-case derate for the SWAP block's own activation transfer.
    swap_eff_h2d, swap_eff_d2h = effective_bw(cfg, hw)

    # Augmented persistent set: prefix | layout.mandatory_persistent (non-block chunks).
    persistent_ids: frozenset[ChunkId] = layout.effective_persistent_ids(cfg.n_persist)
    n_persist_eff = len(persistent_ids)
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist_eff))
    n_nonpersist = max(0, layout.N_chunk - n_persist_eff)
    nonpersist_chunk_ids: list[int] = [
        cid for cid in range(layout.N_chunk) if ChunkId(cid) not in persistent_ids
    ]

    # Cache per-chunk effective bandwidth once per direction; searcher pays O(N_chunk).
    chunk_bw_fwd: list[tuple[float, float] | None] = [None] * layout.N_chunk
    chunk_bw_bwd: list[tuple[float, float] | None] = [None] * layout.N_chunk
    if chunk_bw_table is not None:
        for cid in nonpersist_chunk_ids:
            entry = chunk_bw_table.get(cid)
            if entry is None:
                chunk_bw_fwd[cid] = effective_bw_for_chunk(
                    ChunkId(cid), cfg, hw, layout, block_map, direction="fwd"
                )
                chunk_bw_bwd[cid] = effective_bw_for_chunk(
                    ChunkId(cid), cfg, hw, layout, block_map, direction="bwd"
                )
            else:
                chunk_bw_fwd[cid] = entry[0]
                chunk_bw_bwd[cid] = entry[1]
    else:
        for cid in nonpersist_chunk_ids:
            chunk_bw_fwd[cid] = effective_bw_for_chunk(
                ChunkId(cid), cfg, hw, layout, block_map, direction="fwd"
            )
            chunk_bw_bwd[cid] = effective_bw_for_chunk(
                ChunkId(cid), cfg, hw, layout, block_map, direction="bwd"
            )

    # NCCL gather + reduce both charged per Eq. 6; reduce is uniform across cached/uncached.
    # Fail-closed on world-mismatched ZeRO-3 traces — silently zeroing under-prices candidates.
    if hw.zero3_shard and hw.gpu_count > 1 and trace.world != hw.gpu_count:
        return _INF_COMPONENTS
    if not hw.zero3_shard or hw.gpu_count <= 1 or trace.world <= 1:
        nccl_gather = 0.0
        nccl_reduce = 0.0
    else:
        # Multi-rank ZeRO-3: fail-closed on missing tables so searcher refreshes.
        if not trace.nccl_gather_s or not trace.nccl_reduce_s:
            return _INF_COMPONENTS
        nccl_gather = _pick_nccl(trace.nccl_gather_s, layout.S_chunk)
        nccl_reduce = _pick_nccl(trace.nccl_reduce_s, layout.S_chunk)
        if nccl_gather <= 0.0 or nccl_reduce <= 0.0:
            return _INF_COMPONENTS

    # Per-chunk comm costs: forward gathers; backward splits cached vs uncached.
    t_fwd_comm_per_chunk: list[float] = [0.0] * layout.N_chunk
    t_bwd_comm_per_chunk_cached: list[float] = [0.0] * layout.N_chunk
    t_bwd_comm_per_chunk_uncached: list[float] = [0.0] * layout.N_chunk
    for cid in nonpersist_chunk_ids:
        eff_h2d_fwd, eff_d2h_fwd = chunk_bw_fwd[cid]  # type: ignore[misc]
        eff_h2d_bwd, eff_d2h_bwd = chunk_bw_bwd[cid]  # type: ignore[misc]
        t_fwd_comm_per_chunk[cid] = _comm_time_chunk(
            layout.S_chunk,
            eff_h2d_fwd,
            eff_d2h_fwd,
            nccl_gather,
            is_backward=False,
            buffer_cached=False,
        )
        t_bwd_comm_per_chunk_cached[cid] = _comm_time_chunk(
            layout.S_chunk,
            eff_h2d_bwd,
            eff_d2h_bwd,
            nccl_gather,
            is_backward=True,
            buffer_cached=True,
            nccl_reduce_s=nccl_reduce,
        )
        t_bwd_comm_per_chunk_uncached[cid] = _comm_time_chunk(
            layout.S_chunk,
            eff_h2d_bwd,
            eff_d2h_bwd,
            nccl_gather,
            is_backward=True,
            buffer_cached=False,
            nccl_reduce_s=nccl_reduce,
        )

    n_block = len(trace.activation_sizes)
    (
        t_fwd_compute_total,
        per_block_compute,
        used_measured,
        t_fwd_compute_base,
    ) = _fwd_compute_time_from_trace(trace, cfg)
    if not used_measured:
        global _WARNED_APPROXIMATE_COMPUTE_PROXY
        if not _WARNED_APPROXIMATE_COMPUTE_PROXY:
            LOG.warning(
                "ProTrain: using approximate compute-rate proxy; re-run profiler "
                "for measured latencies (further occurrences suppressed)"
            )
            _WARNED_APPROXIMATE_COMPUTE_PROXY = True

    # Per-SKU compute-rate calibration; applies to both total and base for bwd fallback.
    sku_scale = _sku_compute_scale(trace, hw)
    if sku_scale != 1.0:
        t_fwd_compute_total *= sku_scale
        t_fwd_compute_base *= sku_scale
        per_block_compute = {bid: v * sku_scale for bid, v in per_block_compute.items()}
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "estimate_runtime: applied per-SKU compute scale %.3f (trace=%s "
                "live_TFLOPS=%.1f trace_TFLOPS=%.1f)",
                sku_scale,
                trace.sku,
                hw.gpu_compute_tflops,
                trace.compute_rate_tflops,
            )
    t_fwd_swap_transfer = 0.0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.SWAP:
            # Worst-case derate; swap-stream competes with chunk prefetch.
            if swap_eff_d2h > 0:
                t_fwd_swap_transfer += act_sz / swap_eff_d2h

    # Phase-2 chunked-wall override gated on cfg.n_swap == 0; SWAP candidates use analytical path.
    fwd_used_phase2_override = False
    bwd_used_phase2_override = False
    if trace.steady_fwd_chunked_wall_s > 0.0 and cfg.n_swap == 0:
        fwd_used_phase2_override = True
        # Translate bootstrap's n_persist using effective sets (prefix | mandatory).
        n_persist_eff_bootstrap = len(
            layout.effective_persistent_ids(trace.phase2_n_persist)
        )
        delta_persist_fwd = max(0, n_persist_eff - n_persist_eff_bootstrap)
        fwd_h2d_per_chunk = layout.S_chunk / full_h2d if full_h2d > 0 else 0.0
        fwd_persist_theoretical_per_chunk = nccl_gather + fwd_h2d_per_chunk
        # Cap theoretical save at empirical per-chunk overhead from bootstrap to avoid over-crediting compute-bound chunks.
        n_nonpersist_bootstrap_fwd = max(0, layout.N_chunk - n_persist_eff_bootstrap)
        if n_nonpersist_bootstrap_fwd > 0:
            empirical_fwd_overhead_per_chunk = max(
                0.0,
                (trace.steady_fwd_chunked_wall_s - t_fwd_compute_base)
                / n_nonpersist_bootstrap_fwd,
            )
            # Overlap-aware floor: in the analytical model a non-persistent
            # chunk only contributes max(0, comm - compute) to wall-time when
            # the buffer pool is large enough to pipeline. If the bootstrap
            # measured high per-chunk overhead with a small n_buffer, that
            # overhead reflects *missing* overlap rather than real comm cost,
            # so the credit for adding persistents must not exceed what an
            # overlap-realised plan would save (else the override over-credits
            # compute-bound candidates and ranks fully-resident above offload).
            if layout.N_chunk > 0:
                fwd_compute_per_chunk = t_fwd_compute_base / layout.N_chunk
            else:
                fwd_compute_per_chunk = 0.0
            overlap_aware_fwd_save = max(
                0.0, fwd_persist_theoretical_per_chunk - fwd_compute_per_chunk
            )
            fwd_persist_save_per_chunk = min(
                fwd_persist_theoretical_per_chunk,
                empirical_fwd_overhead_per_chunk,
                overlap_aware_fwd_save,
            )
        else:
            # Degenerate bootstrap (fully persistent); delta is zero anyway.
            fwd_persist_save_per_chunk = fwd_persist_theoretical_per_chunk
        t_fwd_persist_correction = -delta_persist_fwd * fwd_persist_save_per_chunk
        # Fix 2 (defense-in-depth): penalise candidates whose n_buffer drops
        # below the bootstrap's, because the override measured overhead under
        # the bootstrap's pipelining and silently assumes the same overlap.
        buffer_shortfall_fwd = max(0, trace.phase2_n_buffer - n_buffer)
        t_fwd_buffer_shortfall = (
            buffer_shortfall_fwd * fwd_persist_theoretical_per_chunk
        )
        t_fwd = max(
            0.0,
            trace.steady_fwd_chunked_wall_s
            + t_fwd_swap_transfer
            + t_fwd_persist_correction
            + t_fwd_buffer_shortfall,
        )
    else:
        # Per-chunk forward roofline: max(compute, comm) per non-persistent chunk.
        if layout.N_chunk > 0:
            t_fwd_compute_per_chunk = t_fwd_compute_total / layout.N_chunk
        else:
            t_fwd_compute_per_chunk = 0.0

        t_fwd_persistent_chunks = n_persist_eff * t_fwd_compute_per_chunk
        t_fwd_nonpersistent_chunks = 0.0
        for cid in nonpersist_chunk_ids:
            t_fwd_nonpersistent_chunks += max(
                t_fwd_compute_per_chunk, t_fwd_comm_per_chunk[cid]
            )
        t_fwd = (
            t_fwd_persistent_chunks + t_fwd_nonpersistent_chunks + t_fwd_swap_transfer
        )

    # Pass pre-override forward base so bwd fallback ratios don't read the chunked wall.
    t_bwd_compute_base = _bwd_compute_time_from_trace(trace, t_fwd_compute_base, cfg)
    t_bwd_recompute = 0.0
    t_bwd_swap_prefetch = 0.0
    # OFFLOAD gather wall already charged via per-chunk uncached path / chunked wall.
    n_offload_chunks = 0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            t_block = per_block_compute.get(bid, 0.0)
            if t_block <= 0.0:
                t_block = _compute_time(act_sz)
            t_bwd_recompute += t_block
        elif mode is BlockMode.SWAP:
            if swap_eff_h2d > 0:
                t_bwd_swap_prefetch += act_sz / swap_eff_h2d
        elif mode is BlockMode.OFFLOAD:
            # Diagnostic count; backward gather wall is already charged elsewhere.
            n_offload_chunks += sum(
                1
                for cid in layout.block_to_chunks.get(bid, ())
                if ChunkId(int(cid)) not in persistent_ids
            )

    _ = n_offload_chunks

    t_bwd_compute_total = t_bwd_compute_base + t_bwd_recompute
    # Phase-2 override gated on cfg.n_swap == 0; SWAP candidates use analytical path.
    if (
        trace.steady_bwd_chunked_wall_s > 0.0
        and (trace.phase2_n_checkpoint == 0 or trace.phase2_per_block_recompute_s > 0.0)
        and cfg.n_swap == 0
    ):
        bwd_used_phase2_override = True
        # Translate delta_cached against bootstrap so override isn't flat in n_buffer.
        n_nonpersist_bootstrap = max(
            0,
            layout.N_chunk
            - len(layout.effective_persistent_ids(trace.phase2_n_persist)),
        )
        bootstrap_cached = min(trace.phase2_n_buffer, n_nonpersist_bootstrap)
        candidate_cached = min(n_buffer, n_nonpersist)
        delta_cached = candidate_cached - bootstrap_cached
        # Per-hit save: gather + H2D reload skipped. full_h2d for marginal high-index chunks.
        save_h2d_bw = full_h2d
        h2d_save_per_hit = layout.S_chunk / save_h2d_bw if save_h2d_bw > 0 else 0.0
        gather_save_per_hit = nccl_gather + h2d_save_per_hit
        t_bwd_buffer_correction = -delta_cached * gather_save_per_hit

        # n_persist translation: newly-persistent chunks save full uncached cost.
        n_persist_eff_bootstrap_bwd = len(
            layout.effective_persistent_ids(trace.phase2_n_persist)
        )
        delta_persist_bwd = max(0, n_persist_eff - n_persist_eff_bootstrap_bwd)
        bwd_h2d_per_chunk = layout.S_chunk / full_h2d if full_h2d > 0 else 0.0
        bwd_d2h_per_chunk = (
            layout.S_chunk / hw.pcie_d2h_bps if hw.pcie_d2h_bps > 0 else 0.0
        )
        bwd_persist_theoretical_per_chunk = (
            nccl_gather + bwd_h2d_per_chunk + bwd_d2h_per_chunk
        )
        # Cap at empirical per-chunk overhead derived from unwrapped trace fields.
        bootstrap_recompute_bwd = (
            trace.phase2_n_checkpoint * trace.phase2_per_block_recompute_s
        )
        n_nonpersist_bootstrap_bwd = max(
            0, layout.N_chunk - n_persist_eff_bootstrap_bwd
        )
        # Config-independent backward compute floor; prefer unwrapped steady, else fwd*ratio.
        if trace.steady_bwd_wall_s > 0.0:
            bwd_compute_floor = trace.steady_bwd_wall_s
        elif trace.steady_fwd_wall_s > 0.0 and t_fwd_compute_base > 0.0:
            measured_ratio = max(
                1.0,
                min(
                    3.0,
                    trace.steady_bwd_wall_s / trace.steady_fwd_wall_s
                    if trace.steady_bwd_wall_s > 0.0
                    else (
                        1.0
                        if 0.0 < trace.trainable_param_fraction < 0.05
                        else _BWD_FWD_COMPUTE_RATIO
                    ),
                ),
            )
            bwd_compute_floor = t_fwd_compute_base * measured_ratio
        else:
            bwd_compute_floor = 0.0
        if n_nonpersist_bootstrap_bwd > 0:
            empirical_bwd_overhead_per_chunk = max(
                0.0,
                (
                    trace.steady_bwd_chunked_wall_s
                    - bwd_compute_floor
                    - bootstrap_recompute_bwd
                )
                / n_nonpersist_bootstrap_bwd,
            )
            # Overlap-aware floor mirrors the forward branch: cap the credit
            # at max(0, comm - compute) per chunk so the override never
            # double-counts a save the analytical overlap model would have
            # absorbed for free.
            if layout.N_chunk > 0:
                bwd_compute_per_chunk = bwd_compute_floor / layout.N_chunk
            else:
                bwd_compute_per_chunk = 0.0
            overlap_aware_bwd_save = max(
                0.0, bwd_persist_theoretical_per_chunk - bwd_compute_per_chunk
            )
            bwd_persist_save_per_chunk = min(
                bwd_persist_theoretical_per_chunk,
                empirical_bwd_overhead_per_chunk,
                overlap_aware_bwd_save,
            )
        else:
            bwd_persist_save_per_chunk = bwd_persist_theoretical_per_chunk
        t_bwd_persist_correction = -delta_persist_bwd * bwd_persist_save_per_chunk
        # NOTE: backward branch does NOT add a separate buffer-shortfall surcharge.
        # Unlike the forward branch (which starts from steady_fwd_chunked_wall_s
        # without an explicit cache-delta term), the backward branch already
        # converts a smaller n_buffer into a positive surcharge via
        # t_bwd_buffer_correction = -delta_cached * gather_save_per_hit
        # (delta_cached < 0 when n_buffer < phase2_n_buffer). Adding a symmetric
        # buffer_shortfall term here would charge the same shortage twice and
        # systematically over-price low-buffer candidates, mis-ranking the search.
        t_bwd = max(
            0.0,
            t_bwd_compute_total
            + t_bwd_swap_prefetch
            + t_bwd_buffer_correction
            + t_bwd_persist_correction,
        )
    else:
        if layout.N_chunk > 0:
            t_bwd_compute_per_chunk = t_bwd_compute_total / layout.N_chunk
        else:
            t_bwd_compute_per_chunk = 0.0

        # Cached chunks are the suffix of nonpersist_chunk_ids (LRU survivors of forward).
        n_cached = min(n_buffer, n_nonpersist)
        cached_ids: set[int] = (
            set(nonpersist_chunk_ids[-n_cached:]) if n_cached > 0 else set()
        )

        # Persistent chunks: only reduce_scatter contributes to comm (Eq. 6 first branch).
        t_bwd_persistent_chunks = n_persist_eff * max(
            t_bwd_compute_per_chunk, nccl_reduce
        )
        t_bwd_nonpersistent_chunks = 0.0
        for cid in nonpersist_chunk_ids:
            if cid in cached_ids:
                comm = t_bwd_comm_per_chunk_cached[cid]
            else:
                comm = t_bwd_comm_per_chunk_uncached[cid]
            t_bwd_nonpersistent_chunks += max(t_bwd_compute_per_chunk, comm)
        t_bwd = (
            t_bwd_persistent_chunks + t_bwd_nonpersistent_chunks + t_bwd_swap_prefetch
        )

    # Model-state bytes per chunk; fp16 params-only fallback when trace field missing.
    _MS_PER_CHUNK_FLOOR = 1.0  # bytes — guard against zero-rate divides
    model_state_total = int(getattr(trace, "model_state_bytes", 0) or 0)
    if model_state_total <= 0:
        fp16_total = layout.N_chunk * layout.S_chunk
        if fp16_total > 0:
            global _WARNED_MODEL_STATE_MISSING
            if not _WARNED_MODEL_STATE_MISSING:
                LOG.warning(
                    "estimate_runtime: trace.model_state_bytes is missing or "
                    "zero (%d); falling back to fp16 params-only total "
                    "%dB (N_chunk=%d * S_chunk=%d). Optimizer-step costs "
                    "will UNDER-count full Adam state — refresh the profiler "
                    "trace cache to restore fidelity.",
                    model_state_total,
                    fp16_total,
                    layout.N_chunk,
                    layout.S_chunk,
                )
                _WARNED_MODEL_STATE_MISSING = True
        model_state_total = fp16_total
    if layout.N_chunk > 0:
        ms_per_chunk = max(model_state_total / layout.N_chunk, _MS_PER_CHUNK_FLOOR)
    else:
        ms_per_chunk = 0.0

    # cpu_adam_bytes_per_sec=0 sentinel: optim_wrapper skips CPU step; t_cpu_optim → 0.
    if hw.cpu_adam_bytes_per_sec > 0.0:
        cpu_adam_bps = hw.cpu_adam_bytes_per_sec
    else:
        global _WARNED_CPU_ADAM_UNAVAILABLE
        if not _WARNED_CPU_ADAM_UNAVAILABLE:
            LOG.warning(
                "estimate_runtime: cpu_adam_bytes_per_sec=0 — treating CPU "
                "Adam as unavailable (matches optim_wrapper's cpu_optim=None "
                "path). Non-persistent chunks contribute 0 to t_cpu_optim. "
                "Note that under this state non-persistent chunks are NOT "
                "actually being stepped at runtime either; install/fix "
                "DeepSpeed for full coverage."
            )
            _WARNED_CPU_ADAM_UNAVAILABLE = True
        cpu_adam_bps = 0.0  # sentinel — t_cpu_optim collapses to 0

    if hw.gpu_adam_bytes_per_sec > 0.0:
        gpu_adam_bps = hw.gpu_adam_bytes_per_sec
    else:
        global _WARNED_GPU_ADAM_FALLBACK
        if not _WARNED_GPU_ADAM_FALLBACK:
            LOG.warning(
                "estimate_runtime: gpu_adam_bytes_per_sec unavailable; using "
                "fallback %.2e (re-run profiler for a calibrated rate)",
                _GPU_ADAM_FALLBACK,
            )
            _WARNED_GPU_ADAM_FALLBACK = True
        gpu_adam_bps = _GPU_ADAM_FALLBACK

    t_gpu_optim = n_persist_eff * ms_per_chunk / gpu_adam_bps
    # ZeRO-3 divides per-chunk CPU-Adam by world_size; replicated DDP doesn't.
    cpu_shard_divisor = max(1, hw.gpu_count) if hw.zero3_shard else 1
    if cpu_adam_bps <= 0.0:
        # CPU Adam unavailable: mark configs that offload as infeasible.
        if n_nonpersist > 0:
            return _INF_COMPONENTS
        t_cpu_optim = 0.0
    else:
        t_cpu_optim = n_nonpersist * (ms_per_chunk / cpu_shard_divisor) / cpu_adam_bps

    # n_persist corrections live inline in the fwd/bwd override branches above.
    # CPU/GPU optimizers are step-boundary work now; they may overlap each other,
    # but not the already-finished backward pass.
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "estimate_runtime_components: cfg=%s t_fwd=%.4fs t_bwd=%.4fs "
            "t_gpu_opt=%.4fs t_cpu_opt=%.4fs",
            cfg,
            t_fwd,
            t_bwd,
            t_gpu_optim,
            t_cpu_optim,
        )
    _ = n_block
    return (
        t_fwd,
        t_bwd,
        t_gpu_optim,
        t_cpu_optim,
        fwd_used_phase2_override,
        bwd_used_phase2_override,
    )


__all__ = ["estimate_runtime"]
