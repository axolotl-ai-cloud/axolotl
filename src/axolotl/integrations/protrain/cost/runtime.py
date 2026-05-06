"""Runtime (wall-clock) cost estimator for the ProTrain searcher (§3.3, App A.1).

Implements the per-chunk runtime model from the paper. The communication
sub-terms map directly onto numbered equations; the compute and optimizer
sub-terms are described in prose in App A.1 but not numbered:

    T_iter    = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)      [Eq. 2]
    T_fwd     = sum_chunks  max(T_compute_chunk, T_comm_chunk)     [Eq. 3]
    T_bwd     = sum_chunks  max(T_compute_chunk + T_recomp_chunk,
                                T_comm_chunk)                      [Eq. 5]
    T_FWD-prefetch_comm    (per-chunk, fwd)                        [Eq. 4]
    T_reduce-offload_comm  (per-chunk, bwd, non-persistent)        [Eq. 6]
    T_BWD-prefetch_comm    (per-chunk, bwd, evicted-from-buffer)   [Eq. 7]
    T_gpu_opt = sum_{persistent chunks} T_step(chunk)              [App A.1, prose]
    T_cpu_opt = sum_{non-persistent chunks} T_step(chunk)          [App A.1, prose]

Key accounting rules (summary §3.3, paper §3.3.1):

- Persistent chunks contribute no prefetch/gather/H2D/D2H cost (they
  never leave GPU), but in ZeRO-3 mode they still pay the per-chunk
  ``T_reduce`` (reduce-scatter) at the end of backward — paper Eq. 6
  charges ``T_reduce`` on every chunk ``i ≤ N_chunk``.
- Buffer-cached chunks skip re-gather in backward but still pay
  ``T_reduce`` and the D2H grad-offload — gather is amortised by the
  cache; reduce always happens when the chunk's grads finalize.
- CPU-Adam overlaps GPU backward; only exposed if ``T_cpu_optim`` exceeds
  ``T_bwd + T_gpu_optim``.
- CKPT blocks add a recomputation-compute term to backward.
- SWAP blocks add CPU<->GPU activation transfer on both sides.
- For single-rank (``world == 1``) or replicated layouts
  (``zero3_shard=False``) the NCCL gather/reduce terms are 0 because
  there are no per-chunk collectives.

PCIe bandwidth contention (§3.3 / §A.1):

- Per paper §3.3, the cost model "estimates the swapping time, identifies
  the affected chunks, and uses the reduced bandwidth instead." The
  identification step is now per-chunk via
  :func:`bandwidth.chunk_swap_overlap_count`; the reduced bandwidth
  per chunk is :func:`bandwidth.effective_bw_for_chunk`.
- This estimator iterates non-persistent chunks once per direction
  (forward + backward), looks up each chunk's effective bandwidth
  (full PCIe when its prefetch window overlaps no SWAP block,
  derated proportionally to the overlap count otherwise), and sums
  the per-chunk comm cost. The earlier four-way split (cached vs.
  uncached × affected vs. unaffected) is gone — replaced by a single
  per-chunk loop that naturally handles both axes (the cached/
  uncached predicate is a simple ``cid >= n_chunk - n_buffer``
  check applied inside the loop).
- The Eq. 6 ``T_reduce`` term is a pure NCCL collective (no PCIe
  involved), so the per-chunk derate does not touch it. It is
  charged uniformly at the trace-measured collective time.
- Caching: ``effective_bw_for_chunk`` is called once per
  (chunk, direction) and the result memoised inline (the per-chunk
  vector is materialised once per ``estimate_runtime`` invocation),
  so the searcher's enumeration loop pays only ``O(N_chunk)`` for
  the contention model, not ``O(N_chunk * N_block)``.

The estimator is a pure function of the frozen dataclass inputs; it does
not allocate tensors or touch CUDA.
"""

from __future__ import annotations

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

# FALLBACK compute throughput proxy — only used when the ProfilerTrace has no
# ``op_latencies`` (e.g. a trace recorded on CPU, or a stale cached trace from
# before TRACE_VERSION=2). When measured per-op latencies ARE available, the
# cost model consumes them directly and this constant is not read.
_COMPUTE_BYTES_PER_SEC: float = 3.0e11  # ~300 GB/s, rough 3090 effective

# Fallback CPU-Adam step throughput (bytes of optim-state processed per
# second). The cost model prefers the MEASURED rate from
# ``HardwareProfile.cpu_adam_bytes_per_sec`` (populated by
# ``profiler/hw_bench.measure_cpu_adam``); this constant is only consumed
# when the measurement returned 0.0 (e.g. DeepSpeedCPUAdam failed to
# compile, common on dev rigs with CUDA toolchain mismatches).
# DeepSpeedCPUAdam benches around 1-2 GB/s per step on a decent Xeon/
# Threadripper; the "20 B/param" accounting in hw_bench pushes the
# measured throughput a bit higher — 8 GB/s is a reasonable middle-of-
# the-road prior that avoids under- or over-predicting catastrophically.
_CPU_ADAM_FALLBACK: float = 8.0e9

# Fallback GPU FusedAdam throughput, same semantics as ``_CPU_ADAM_FALLBACK``.
# GPU Adam is HBM-bandwidth-bound on 3090s; 500 GB/s is a mid-range prior
# that matches the 3090's sustained HBM BW.
_GPU_ADAM_FALLBACK: float = 5.0e11

# Backward-vs-forward compute ratio when the trace has forward latencies but
# no per-block backward split. The synthetic ``<backward>`` op records a
# single aggregate latency; using that directly is more accurate than the
# heuristic factor, and the code below prefers it when present.
_BWD_FWD_COMPUTE_RATIO: float = 2.0

# Clamp bounds for the hook-less / hooked forward wall-time calibration
# scale (see ``_hook_scale_factor``). An absurdly small scale (< 0.3) would
# over-correct the per-block sum into unrealistic territory; a scale > 1.0
# means "hooked forward was FASTER than un-hooked", which should not happen
# on any well-formed trace (the hook path strictly adds work). Both cases
# indicate a measurement glitch — clamp and WARN instead of propagating.
_HOOK_SCALE_MIN: float = 0.3
_HOOK_SCALE_MAX: float = 1.0

# Clamp bounds for the per-SKU compute-rate calibration scale. The 3090 vs
# 3090 Ti compute spread on a 4K fp16 GEMM is ~5-10%; bigger ratios (e.g.
# 0.5 or 2.0) almost certainly indicate a measurement glitch (cold cuBLAS
# handle, thermal throttling on one of the cards, etc.) rather than a real
# SKU difference, and applying them would distort predictions more than
# leaving them at 1.0. Clamp + WARN.
_SKU_SCALE_MIN: float = 0.5
_SKU_SCALE_MAX: float = 2.0


def _sku_compute_scale(trace: ProfilerTrace, hw: HardwareProfile) -> float:
    """Return the trace-vs-live compute-rate ratio, clamped.

    Cached traces capture ``compute_rate_tflops`` on the trace SKU; the
    live HardwareProfile carries ``gpu_compute_tflops`` for the device the
    searcher is currently planning for. When both are non-zero, this
    function returns ``trace.compute_rate_tflops / hw.gpu_compute_tflops``
    — the factor the cost model multiplies into per-op forward time so a
    trace from a faster card predicts a slower iter on a slower card and
    vice versa.

    Identity (1.0) is returned when either side is unmeasured (pre-v8
    cache, hw_bench measurement glitch). The clamp keeps a single noisy
    measurement from blowing the prediction up — the noise floor on the
    GEMM bench is ~2%, so 0.5/2.0 bounds are extremely loose.
    """
    if trace.compute_rate_tflops <= 0.0 or hw.gpu_compute_tflops <= 0.0:
        return 1.0
    raw = trace.compute_rate_tflops / hw.gpu_compute_tflops
    if raw < _SKU_SCALE_MIN or raw > _SKU_SCALE_MAX:
        LOG.warning(
            "SKU compute-rate scale out of sane range (%.3f = trace %.1f / "
            "live %.1f TFLOPS); clamping to [%.2f, %.2f]. Treat with "
            "suspicion — likely a measurement glitch on one of the two SKUs.",
            raw,
            trace.compute_rate_tflops,
            hw.gpu_compute_tflops,
            _SKU_SCALE_MIN,
            _SKU_SCALE_MAX,
        )
    return max(_SKU_SCALE_MIN, min(_SKU_SCALE_MAX, raw))


def _hook_scale_factor(trace: ProfilerTrace) -> float:
    """Return the steady/hooked forward wall-time ratio, clamped to a sane range.

    The profiler records both a ``hooked_fwd_wall_s`` (total wall-clock of
    the hooked forward pass — inflated by pre/post forward hook dispatch)
    and a ``steady_fwd_wall_s`` (the same forward, timed BEFORE hooks were
    installed). On transformer-sized models the ratio lands around 0.3-0.5
    (i.e. the hooked pass is 2-3x slower than steady-state), and that
    ratio is the scalar correction the cost model needs to apply to the
    hooked per-op latencies when predicting steady-state ``t_fwd``.

    Backward compatibility: traces older than ``TRACE_VERSION=4`` have
    both fields at 0.0 — this function returns 1.0 (identity) for those,
    matching pre-calibration behavior. No warning is logged to keep
    legacy traces quiet; the cache-version bump is the corrective path.
    """
    if trace.hooked_fwd_wall_s <= 0.0 or trace.steady_fwd_wall_s <= 0.0:
        return 1.0
    raw = trace.steady_fwd_wall_s / trace.hooked_fwd_wall_s
    if raw > _HOOK_SCALE_MAX or raw < _HOOK_SCALE_MIN:
        LOG.warning(
            "hook-scale ratio out of sane range (%.3f = steady %.4fs / hooked "
            "%.4fs); clamping to [%.2f, %.2f]",
            raw,
            trace.steady_fwd_wall_s,
            trace.hooked_fwd_wall_s,
            _HOOK_SCALE_MIN,
            _HOOK_SCALE_MAX,
        )
    return max(_HOOK_SCALE_MIN, min(_HOOK_SCALE_MAX, raw))


def _compute_time(activation_bytes: int) -> float:
    """Rough compute time proxy — used only as a fallback for traces that
    carry no measured ``op_latencies`` (see ``_fwd_compute_time_from_trace``).
    """
    return activation_bytes / _COMPUTE_BYTES_PER_SEC


def _block_compute_time(trace: ProfilerTrace, block_id: BlockId) -> float:
    """Wall-clock forward compute for one block from profiler measurements.

    Sums the measured op latencies for all forward ops whose ``block_id``
    matches. Returns 0.0 for blocks that have no measured ops (e.g. non-
    block ops like embedding) — the caller is responsible for handling
    that case with a fallback.
    """
    total_s = 0.0
    for op in trace.op_order:
        if op.block_id != block_id or not op.is_forward:
            continue
        total_s += trace.op_latencies.get(op.op_id, 0.0)
    return total_s


def _fwd_compute_time_from_trace(
    trace: ProfilerTrace,
) -> tuple[float, dict[BlockId, float], bool]:
    """Return (total_fwd_compute_s, per_block_compute_s, used_measured).

    Preference order (highest first):

    1. **Phase-2 chunked forward measurement** (TRACE_VERSION ≥ 11): if
       ``steady_fwd_chunked_wall_s > 0``, return it as the forward
       total. The per-block distribution comes from the per-op path
       (used by ``estimate_runtime`` for CKPT recompute accounting and
       the per-chunk roofline split). Forward is approximately
       config-independent at the cost-model level (no recompute on
       forward; differences in n_persist / n_buffer between bootstrap
       and candidate change comm overlap marginally), so the
       measurement applies as the new baseline for ANY candidate cfg
       the search evaluates.
    2. **Per-op-latency sum + hook-scale + roofline cap** (TRACE_VERSION
       ≥ 2): if the trace carries ``op_latencies``, apply the
       hook-dispatch calibration scale (``steady_fwd_wall_s /
       hooked_fwd_wall_s``, clamped to ``[_HOOK_SCALE_MIN,
       _HOOK_SCALE_MAX]``) to the per-op sum. On transformer-sized
       models this strips ~2.5-8x hook inflation from the measurement.
       The scaled total is then capped at ``steady_fwd_wall_s`` (or 2x
       activation-byte roofline as a legacy fallback) to protect
       against runaway measurements on stale traces.
    3. **Activation-size roofline** (always available): pure fallback
       for traces with no measured latencies; returns
       ``used_measured=False``.

    Mirrors the precedence pattern of
    :func:`_bwd_compute_time_from_trace` (phase-2 chunked > steady
    unwrapped > heuristic), with the simplification that forward needs
    no per-cfg adjustment because it doesn't recompute.
    """
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

        # PRIMARY correction: apply the clamped hook-dispatch scale.
        # Legacy (pre-v4) traces have 0.0 wall-times — the scale function
        # returns 1.0 (identity) in that case, matching old behavior.
        scale = _hook_scale_factor(trace)
        per_block = {bid: v * scale for bid, v in hooked_per_block.items()}
        total = hooked_total * scale

        if total > 0.0:
            # SECONDARY safety: cap absolute magnitude. Two upper bounds
            # in priority order:
            #   (a) measured `steady_fwd_wall_s` — the ground-truth
            #       hook-less forward wall; if present, this IS what
            #       steady-state training actually spends on forward.
            #   (b) 2x activation-byte roofline — fallback for legacy
            #       traces (pre-TRACE_VERSION=4) that lack the measurement.
            # Without the cap the searcher reorders toward
            # offload-everything configs that are worse in reality.
            # Preserves per-block SHAPE of the measurement.
            cap = 0.0
            if trace.steady_fwd_wall_s > 0.0:
                cap = trace.steady_fwd_wall_s
            elif roofline_total > 0.0:
                cap = 2.0 * roofline_total
            if cap > 0.0 and total > cap:
                safety = cap / total
                per_block = {bid: v * safety for bid, v in per_block.items()}
                total = cap
            # PHASE-2 FORWARD OVERRIDE (TRACE_VERSION ≥ 11): override
            # the per-op-derived total with the chunked-runtime
            # measurement when populated. Mirrors the precedence
            # pattern in ``_bwd_compute_time_from_trace``. The
            # per-block distribution stays at the per-op-derived shape
            # (used for CKPT recompute accounting); only the total is
            # replaced.
            #
            # Note: the actual t_fwd assembly in ``estimate_runtime``
            # consumes ``trace.steady_fwd_chunked_wall_s`` directly as
            # t_fwd (skipping the per-chunk roofline) because feeding
            # the chunked wall through the per-chunk max(compute,
            # comm) roofline still overshoots reality — the chunked
            # measurement already accounts for chunk-prefetch /
            # gather overlap that the per-chunk roofline assumes
            # unconditionally non-overlapping. Returning the chunked
            # wall as the total here keeps this helper's contract
            # consistent with ``_bwd_compute_time_from_trace`` and
            # makes any downstream consumer that asks "what's the
            # forward compute total?" see the ground-truth
            # measurement.
            if trace.steady_fwd_chunked_wall_s > 0.0:
                total = trace.steady_fwd_chunked_wall_s
            return total, per_block, True

    # Fallback: pure roofline. No measurements available (empty op_latencies).
    return roofline_total, roofline_per_block, False


def _bwd_compute_time_from_trace(trace: ProfilerTrace, t_fwd_total: float) -> float:
    """Return the aggregate backward compute time in seconds.

    Preference order:

    1. **Phase-2 chunked measurement** (TRACE_VERSION ≥ 10): if
       ``steady_bwd_chunked_wall_s > 0`` AND ``phase2_per_block_recompute_s > 0``,
       use the chunked measurement minus the bootstrap's recompute term.
       This returns the **base** backward time (no recompute) — the
       caller then adds the candidate ``block_map``'s recompute on top
       in the same way as the v8 path. The translation is:

           base_bwd = steady_bwd_chunked_wall_s
                    - phase2_n_checkpoint * phase2_per_block_recompute_s

       (clamped to ≥ 0 for numerical safety; a base of 0 means the
       measured chunked time was entirely recompute, which only happens
       when the bootstrap had every block CKPT'd and the model was
       essentially all-recompute already. Caller's per-cfg recompute
       term still adds the right amount on top.)

    2. **Steady (unwrapped) measurement** (TRACE_VERSION ≥ 7): measured
       ``steady_bwd_wall_s / steady_fwd_wall_s`` ratio from the 4-iter
       hot loop. Captures the actual transformer-specific bwd/fwd
       relationship on the measured hardware — typically 1.5-2.2x
       depending on the attention implementation. Used when phase-2
       didn't run (smaller models where the unwrapped backward fits)
       and is more accurate than the heuristic.

    3. **Heuristic** (always available): trainable-fraction-aware.
       LoRA / adapter training has ~0.1% trainable; backward only flows
       through those params, ratio ≈ 1.0. Full finetune sees the
       canonical 2.0x. This is the path 7B-LoRA traces hit before
       phase-2 because the unwrapped backward OOMs and the chunked
       measurement hadn't been wired up.

    The hooked aggregate ``<backward>`` latency retained in
    ``trace.op_latencies`` is NOT used — autograd holds the hook-saved
    tensors during the forward which materially distorts the hooked
    backward timing.
    """
    # ---- Path 1: phase-2 chunked measurement ----
    # Gate accepts phase-2 measurements when the chunked backward wall is
    # populated AND we can correctly translate out the bootstrap's recompute:
    #   - bootstrap with ``n_checkpoint > 0`` requires
    #     ``per_block_recompute_s > 0`` to subtract the right amount, OR
    #   - bootstrap with ``n_checkpoint == 0`` is also valid: there was no
    #     recompute to subtract (``per_block_recompute_s`` is naturally 0
    #     in that case), and the chunked wall IS the base backward time.
    # Pre-fix this branch required ``per_block_recompute_s > 0`` and
    # silently rejected ``n_checkpoint=0`` bootstraps even though their
    # measurement is the cleanest possible base (no recompute baked in).
    if trace.steady_bwd_chunked_wall_s > 0.0 and (
        trace.phase2_n_checkpoint == 0 or trace.phase2_per_block_recompute_s > 0.0
    ):
        bootstrap_recompute = (
            trace.phase2_n_checkpoint * trace.phase2_per_block_recompute_s
        )
        base = max(0.0, trace.steady_bwd_chunked_wall_s - bootstrap_recompute)
        return base
    # ---- Path 2: steady unwrapped measurement ----
    if trace.steady_bwd_wall_s > 0.0 and trace.steady_fwd_wall_s > 0.0:
        measured_ratio = trace.steady_bwd_wall_s / trace.steady_fwd_wall_s
        # Clamp to a sane range — if the measurement is wildly off
        # (measurement noise or forward OOM that fell through), don't
        # let it propagate. Transformers run between 1.0x (LoRA, autograd
        # skips frozen subgraphs) and 3x (full-finetune with attention recomp).
        measured_ratio = max(1.0, min(3.0, measured_ratio))
        return t_fwd_total * measured_ratio
    # ---- Path 3: trainable-fraction-aware heuristic ----
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
    """Return the communication time for a single non-persistent chunk.

    Three-way split on (is_backward, buffer_cached):

    - Forward (any chunk): NCCL gather + PCIe H2D (CPU->GPU shard reload)
      to populate the chunk buffer before compute. ``nccl_reduce_s`` is
      ignored on forward.
    - Backward, buffer-cached: the buffer still has the chunk from
      forward, so the all-gather is skipped and the H2D reload is also
      skipped — only the PCIe D2H (grad reduce-offload) remains. The
      ZeRO-3 reduce-scatter (``nccl_reduce_s``) IS charged here per
      paper Eq. 6: reduce always happens when the chunk's grads
      finalize, regardless of whether the chunk was buffer-cached
      (gather is amortized by the cache; reduce is not).
    - Backward, uncached: the chunk was evicted from the buffer pool
      between forward and backward (n_buffer < n_nonpersist), so the
      shard must be re-fetched H2D *before* the all-gather can run, then
      the grad is drained D2H after the backward op. Cost is
      ``collective + reduce + S_chunk/eff_h2d + S_chunk/eff_d2h``.

    The third case was previously charging only ``collective +
    S_chunk/eff_d2h`` and so systematically undercosted OFFLOAD / low-
    ``n_buffer`` configs (CodeRabbit Round-5 R5-B). The fix here applies
    to non-persistent chunks evicted between forward and backward.
    OFFLOAD blocks reuse this same per-chunk uncached gather event for
    the saved-tensor unpack rebind (one gather populates the chunk
    buffer; the autograd unpack hook rebinds saved-tensor views into
    that same buffer in-step), so this branch is the single source of
    truth for the OFFLOAD backward gather wall. An earlier revision
    added a separate ``T_bwd_gather`` term in
    :func:`estimate_runtime`, but that double-counted the gather (CR
    PR #13 Round-2 R3186562956); the explicit term has been removed
    and the per-chunk uncached cost here charges it exactly once.

    ``nccl_reduce_s`` (default 0.0) is the per-chunk reduce-scatter
    collective time at this S_chunk payload; the caller pre-selects the
    right entry from ``trace.nccl_reduce_s`` and passes 0.0 when there
    is no collective (single-rank, ``zero3_shard=False``, etc.). Added
    uniformly to BOTH backward branches per paper Eq. 6.
    """
    # NCCL gather contribution is size-dependent; the trace keys
    # ``nccl_gather_s`` by payload bytes. We pre-selected the right
    # entry in the caller.
    collective = nccl_gather_s

    # Defensive divisions: a pathological/unmeasured eff_*2d collapses
    # the corresponding PCIe term to 0 instead of raising.
    h2d = S_chunk / eff_h2d if eff_h2d > 0 else 0.0
    d2h = S_chunk / eff_d2h if eff_d2h > 0 else 0.0

    if not is_backward:
        # Forward: gather then H2D reload to populate the chunk buffer.
        # No reduce on forward.
        return collective + h2d
    if buffer_cached:
        # Backward cache-hit: skip both the all-gather and the H2D
        # reload; the grad drain plus reduce-scatter remain.
        return d2h + nccl_reduce_s
    # Backward uncached: evicted-from-buffer chunk needs H2D reload
    # before the gather, plus the D2H grad-offload after compute, plus
    # reduce-scatter for the chunk's grad shard.
    return collective + h2d + d2h + nccl_reduce_s


def _pick_nccl(nccl_table: dict, payload_bytes: int) -> float:
    """Look up the nearest payload size in an NCCL latency table.

    ``nccl_table`` is ``{payload_bytes -> seconds}``. If empty, return
    0.0 — single-rank / no-collective case.
    """
    if not nccl_table:
        return 0.0
    # Nearest-size lookup in log space would be fancier; cheapest
    # correct thing is pick the entry whose key is closest.
    best = min(nccl_table.keys(), key=lambda k: abs(int(k) - payload_bytes))
    return float(nccl_table[best])


def estimate_runtime(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,
) -> float:
    """Estimate wall-clock iteration time in seconds.

    See module docstring for the equations and accounting rules.
    """
    # Per-chunk timeline-overlap bandwidth (paper §3.3 — "identify the
    # affected chunks, and use the reduced bandwidth instead"):
    # :func:`effective_bw_for_chunk` returns the per-chunk
    # ``(eff_h2d, eff_d2h)`` pair derated only for chunks whose
    # prefetch window overlaps a SWAP block. We materialise the
    # per-chunk bandwidth vectors once per direction below and feed
    # them into the per-chunk comm sums.
    full_h2d = hw.pcie_h2d_bps

    # Legacy worst-case derate, used for the SWAP block's own
    # activation D2H/H2D (the swap stream's per-block transfer cost).
    # The per-chunk model derates chunk prefetch when it overlaps swap
    # traffic; symmetrically, the swap transfer itself runs against
    # concurrent chunk prefetch on the prefetch stream and is
    # conservatively billed at the worst-case derate. Pre-fix the
    # SWAP block transfer used the same scalar derate as the per-chunk
    # cost; preserving that here keeps the SWAP-block accounting
    # untouched while the chunk cost moves to the per-chunk model.
    swap_eff_h2d, swap_eff_d2h = effective_bw(cfg, hw)

    # ----- Per-chunk comm / compute decomposition -----------------------
    n_persist = max(0, min(cfg.n_persist, layout.N_chunk))
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist))
    n_nonpersist = max(0, layout.N_chunk - n_persist)

    # Materialise per-chunk effective bandwidth vectors. Cached once
    # here (NOT inside ``_comm_time_chunk``) so the searcher's
    # enumeration loop pays O(N_chunk) per direction, not
    # O(N_chunk * N_block). Persistent chunks (idx < n_persist) are
    # included as ``None`` placeholders for index alignment but never
    # consulted (persistent chunks have no prefetch traffic). When
    # ``cfg.n_swap == 0`` the helper short-circuits to full bandwidth
    # for every chunk — identical numerics to the pre-contention path.
    chunk_bw_fwd: list[tuple[float, float] | None] = [None] * layout.N_chunk
    chunk_bw_bwd: list[tuple[float, float] | None] = [None] * layout.N_chunk
    for cid in range(n_persist, layout.N_chunk):
        chunk_bw_fwd[cid] = effective_bw_for_chunk(
            ChunkId(cid), cfg, hw, layout, block_map, direction="fwd"
        )
        chunk_bw_bwd[cid] = effective_bw_for_chunk(
            ChunkId(cid), cfg, hw, layout, block_map, direction="bwd"
        )

    # NCCL table lookup at chunk-payload size. Single-rank -> world==1
    # and the tables should be empty (or contain zero times), yielding
    # 0s here.
    #
    # Both ``nccl_gather`` (all-gather, populates the chunk buffer
    # before compute) and ``nccl_reduce`` (reduce-scatter, drains the
    # chunk's grad shard at the end of backward) are charged per paper
    # Eq. 6. The earlier model dropped ``T_reduce`` entirely, justified
    # as "overlapped with compute under ZeRO-3"; that overlap is at
    # best partial on PCIe Gen3-class fabrics and is not a modelling
    # assumption the paper makes. Charging ``T_reduce`` in full here is
    # conservative — the cost model now reflects Eq. 6 directly and
    # any genuine compute/reduce overlap shows up as estimator
    # over-prediction rather than a structural under-credit on the
    # comm term that lets the searcher pick reduce-heavy configs the
    # runtime can't actually overlap.
    #
    # Important: ``T_reduce`` is added UNIFORMLY to every backward
    # chunk (persistent + cached + uncached non-persistent). It is NOT
    # part of the buffer-cache-hit savings — gather is amortised by
    # the cache (the chunk is already resident), but reduce-scatter
    # always happens when the chunk's grads finalize, regardless of
    # whether the chunk was buffer-cached. The phase-2 buffer-cache
    # delta correction at ~line 741 therefore continues to subtract
    # only ``nccl_gather + h2d`` per delta cache hit; ``nccl_reduce``
    # is invariant in n_buffer and cancels out of the delta.
    if not hw.zero3_shard or hw.gpu_count <= 1 or trace.world <= 1:
        nccl_gather = 0.0
        nccl_reduce = 0.0
    else:
        nccl_gather = _pick_nccl(trace.nccl_gather_s, layout.S_chunk)
        nccl_reduce = _pick_nccl(trace.nccl_reduce_s, layout.S_chunk)

    # Per-chunk comm-cost vectors. Each entry is the
    # :func:`_comm_time_chunk` cost evaluated at the per-chunk
    # effective bandwidth (full PCIe when no SWAP overlap, derated
    # otherwise — see :func:`bandwidth.effective_bw_for_chunk`).
    #
    # Forward: every non-persistent chunk needs gather + H2D
    # (no buffer cache yet). ``nccl_reduce`` is not passed (forward
    # branch of ``_comm_time_chunk`` ignores it).
    #
    # Backward: per-chunk lookup splits into cached vs. uncached based
    # on buffer-pool residency at backward time. Cached chunks (the
    # last ``n_buffer`` non-persistent chunks — see ``n_cached`` below)
    # skip re-gather and the H2D reload; uncached chunks pay the full
    # round-trip. ``nccl_reduce`` is added inside ``_comm_time_chunk``
    # uniformly to BOTH backward branches per paper Eq. 6 (the
    # cancellation invariant referenced in the phase-2 correction at
    # ~line 741 holds chunk-by-chunk).
    t_fwd_comm_per_chunk: list[float] = [0.0] * layout.N_chunk
    t_bwd_comm_per_chunk_cached: list[float] = [0.0] * layout.N_chunk
    t_bwd_comm_per_chunk_uncached: list[float] = [0.0] * layout.N_chunk
    for cid in range(n_persist, layout.N_chunk):
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

    # ----- Forward compute ---------------------------------------------
    # Forward per-block compute is the SUM of measured op latencies for that
    # block when the profiler recorded them; otherwise the activation-size
    # roofline proxy. SWAP blocks add activation H2D/D2H on top of compute.
    n_block = len(trace.activation_sizes)
    t_fwd_compute_total, per_block_compute, used_measured = (
        _fwd_compute_time_from_trace(trace)
    )
    if not used_measured:
        LOG.warning(
            "ProTrain: using approximate compute-rate proxy; re-run profiler "
            "for measured latencies"
        )

    # Per-SKU compute-rate calibration. When the cached trace was captured
    # on a different SKU than the live training device (e.g. trace from
    # 3090 Ti, live 3090), the per-op latencies need to be scaled by the
    # ratio of measured TFLOPS. Same-SKU runs see ratio ≈ 1.0.
    sku_scale = _sku_compute_scale(trace, hw)
    if sku_scale != 1.0:
        t_fwd_compute_total *= sku_scale
        per_block_compute = {bid: v * sku_scale for bid, v in per_block_compute.items()}
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
            # Offload activation CPU-side during forward. Use the
            # legacy worst-case derate — the swap-stream transfer
            # itself competes with concurrent chunk prefetch.
            if swap_eff_d2h > 0:
                t_fwd_swap_transfer += act_sz / swap_eff_d2h

    # PHASE-2 FORWARD OVERRIDE (TRACE_VERSION ≥ 11): when the
    # chunked-runtime forward measurement is available, use it
    # directly as the t_fwd compute+comm baseline rather than
    # re-estimating via the per-chunk roofline. The measurement was
    # captured under a real chunked runtime — gather/prefetch overhead,
    # CPU<->GPU PCIe traffic, NCCL on multi-rank — that the analytical
    # per-chunk max(compute, comm) roofline OVERESTIMATES because the
    # roofline assumes zero comm/compute overlap. The phase-2
    # measurement captures the real overlapping pipeline.
    #
    # SWAP transfer is added on top because phase-2's bootstrap config
    # has n_swap=0 — any candidate using SWAP must pay that activation
    # transfer in addition.
    #
    # SKU compute scale is NOT applied to the chunked wall here —
    # mirrors :func:`_bwd_compute_time_from_trace`, which also
    # consumes ``steady_bwd_chunked_wall_s`` without an SKU scale.
    # The chunked wall already incorporates compute + comm + overlap
    # on the trace SKU; cross-SKU calibration of the chunked
    # measurement requires re-running phase-2 on the new SKU rather
    # than scalar scaling.
    #
    # n_swap gating (paper §3.3 / commit e8f45fd7): the phase-2
    # measurement was captured at ``cfg.n_swap = 0`` (see
    # ``profiler/phase2.py::select_bootstrap_config`` line ~117), so
    # the measured wall reflects forward time WITHOUT any SWAP-stream
    # activation traffic competing with the chunk-prefetch stream.
    # When a candidate is evaluated with ``cfg.n_swap > 0`` the
    # measured wall does not include the per-chunk PCIe contention
    # that swap traffic would actually impose on chunk prefetches —
    # adding ``t_fwd_swap_transfer`` on top covers the swap-stream
    # transfer itself but not the bandwidth derate on the affected
    # chunks. Fall through to the analytical per-chunk path which
    # consults ``chunk_bw_fwd[]`` (built above via
    # :func:`effective_bw_for_chunk`) and applies the chunk-by-chunk
    # SWAP contention model. For ``cfg.n_swap == 0`` candidates
    # (the dominant case on PCIe Gen3 3090s per paper §3.1.2) the
    # phase-2 path stays exactly as before — full speed, no derate.
    if trace.steady_fwd_chunked_wall_s > 0.0 and cfg.n_swap == 0:
        t_fwd = trace.steady_fwd_chunked_wall_s + t_fwd_swap_transfer
    else:
        # Per-chunk forward roofline: max(compute per chunk, comm per chunk).
        # Distribute the per-block compute evenly across non-persistent
        # chunks (persistent chunks are counted in compute but have no
        # comm). This is the chunk-level roofline the paper describes.
        if layout.N_chunk > 0:
            t_fwd_compute_per_chunk = t_fwd_compute_total / layout.N_chunk
        else:
            t_fwd_compute_per_chunk = 0.0

        # Per-chunk loop: forward has only one buffer-state variant
        # (no cache yet — every non-persistent chunk needs to be
        # gathered + H2D reloaded into the buffer before compute), so
        # we just sum ``max(compute, comm[cid])`` over non-persistent
        # chunks, where ``comm[cid]`` is already the per-chunk derated
        # cost from the timeline-overlap model. Persistent chunks pay
        # only compute (no PCIe traffic).
        t_fwd_persistent_chunks = n_persist * t_fwd_compute_per_chunk
        t_fwd_nonpersistent_chunks = 0.0
        for cid in range(n_persist, layout.N_chunk):
            t_fwd_nonpersistent_chunks += max(
                t_fwd_compute_per_chunk, t_fwd_comm_per_chunk[cid]
            )
        t_fwd = (
            t_fwd_persistent_chunks + t_fwd_nonpersistent_chunks + t_fwd_swap_transfer
        )

    # ----- Backward compute --------------------------------------------
    # Baseline backward: either the measured aggregate <backward> latency
    # from the profiler (preferred) or t_fwd * _BWD_FWD_COMPUTE_RATIO. On
    # top of that, CKPT blocks pay one extra forward per CKPT block (their
    # per-block compute time), and SWAP blocks add the activation prefetch.
    t_bwd_compute_base = _bwd_compute_time_from_trace(trace, t_fwd_compute_total)
    t_bwd_recompute = 0.0
    t_bwd_swap_prefetch = 0.0
    # OFFLOAD chunk-gather wall (Option B §4.2) — accounting note.
    #
    # Every non-persistent chunk that is uncached at backward already
    # pays a full backward re-gather: NCCL gather + H2D reload + D2H
    # grad-offload. That cost lives in ``t_bwd_comm_per_chunk_uncached``
    # (the third branch of :func:`_comm_time_chunk`, post CodeRabbit
    # Round-5 R5-B) for the analytical path, and is baked into
    # ``trace.steady_bwd_chunked_wall_s`` for the phase-2 override path
    # (the phase-2 bootstrap is all-CKPT on the same non-persistent
    # layout, so its measured backward wall already contains the gather
    # once per uncached non-persistent chunk).
    #
    # OFFLOAD reuses the same per-chunk gather event for the
    # saved-tensor unpack rebind: the runtime gathers the chunk into
    # the buffer slot exactly once, and the autograd unpack hook
    # rebinds saved-tensor views into that freshly populated buffer in
    # the same step. There is no second collective and no second H2D
    # specific to OFFLOAD beyond what every uncached non-persistent
    # chunk already pays.
    #
    # Pre-fix this estimator added a separate ``t_bwd_gather`` term
    # (``n_offload_chunks * (S_chunk/eff_h2d + nccl_gather)``) on top
    # of both branches, double-counting the gather for OFFLOAD chunks
    # — once via ``t_bwd_comm_per_chunk_uncached`` /
    # ``steady_bwd_chunked_wall_s``, then again as an explicit term —
    # which over-penalised OFFLOAD candidates and pushed the searcher
    # away from the configs Option B is meant to unlock (CodeRabbit
    # PR #13 Round-2 R3186562956). We now charge the gather exactly
    # once via the existing per-chunk uncached path / phase-2 wall and
    # do not add a separate ``t_bwd_gather`` term here.
    #
    # ``n_offload_chunks`` is still computed for diagnostic / memory-
    # accounting symmetry with the (n_checkpoint, n_offload) search
    # axes; the loop also handles CKPT recompute and SWAP prefetch
    # which are unaffected by the dedup.
    n_offload_chunks = 0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            # Recompute the block's forward to restore activations. Use the
            # measured per-block compute when available; fall back to the
            # activation-size proxy for blocks the profiler didn't cover.
            t_block = per_block_compute.get(bid, 0.0)
            if t_block <= 0.0:
                t_block = _compute_time(act_sz)
            t_bwd_recompute += t_block
        elif mode is BlockMode.SWAP:
            # Activation H2D for the SWAP block during backward, on
            # the swap stream. Worst-case derate (same rationale as
            # the forward swap-transfer term above).
            if swap_eff_h2d > 0:
                t_bwd_swap_prefetch += act_sz / swap_eff_h2d
        elif mode is BlockMode.OFFLOAD:
            # Count non-persistent chunks owned by this OFFLOAD block.
            # ``layout.block_to_chunks[bid]`` may contain multiple
            # ChunkIds for wide blocks; persistent chunks (the first
            # ``n_persist``) never leave GPU memory, so they are
            # excluded. The count is retained for diagnostics; its
            # backward gather wall is already charged by
            # ``t_bwd_comm_per_chunk_uncached`` (analytical) or
            # ``steady_bwd_chunked_wall_s`` (phase-2), see the comment
            # block above.
            n_offload_chunks += sum(
                1
                for cid in layout.block_to_chunks.get(bid, ())
                if int(cid) >= n_persist
            )

    # No separate ``t_bwd_gather`` is added — see the OFFLOAD comment
    # block above for the no-double-count argument. Silence the unused
    # ``n_offload_chunks`` count so the diagnostic lifetime is explicit.
    _ = n_offload_chunks

    t_bwd_compute_total = t_bwd_compute_base + t_bwd_recompute
    # Gate mirrors ``_bwd_compute_time_from_trace`` Path 1: accept the
    # chunked measurement when the bootstrap had no CKPT
    # (``per_block_recompute_s`` is naturally 0 there) OR when both fields
    # are populated. Keeps the two consumers of ``steady_bwd_chunked_wall_s``
    # in lock-step on which traces qualify.
    #
    # n_swap gating (paper §3.3 / commit e8f45fd7): same rationale as
    # the forward override above. The phase-2 backward measurement was
    # captured at ``cfg.n_swap = 0`` (see
    # ``profiler/phase2.py::select_bootstrap_config`` line ~117), so it
    # does not include the per-chunk PCIe contention that swap traffic
    # imposes on backward chunk prefetches. Candidates with
    # ``cfg.n_swap > 0`` route to the analytical per-chunk path below
    # which consults ``chunk_bw_bwd[]`` and applies the per-chunk
    # contention derate. ``cfg.n_swap == 0`` stays on the phase-2 path
    # — identical numerics to before.
    if (
        trace.steady_bwd_chunked_wall_s > 0.0
        and (trace.phase2_n_checkpoint == 0 or trace.phase2_per_block_recompute_s > 0.0)
        and cfg.n_swap == 0
    ):
        # PHASE-2 BACKWARD OVERRIDE (TRACE_VERSION >= 10): the chunked
        # backward wall already includes the measured chunk runtime and its
        # real comm/compute overlap. After translating out the bootstrap
        # recompute and adding this candidate's recompute, consume it
        # directly instead of re-injecting analytical per-chunk comm.
        #
        # n_buffer translation (paper §3.3.1 / §4.2):
        # ``t_bwd_compute_total`` already encodes the bootstrap config's
        # cache-hit savings via the measured ``steady_bwd_chunked_wall_s``.
        # When the candidate ``n_buffer`` differs from the bootstrap's
        # ``phase2_n_buffer``, the candidate gets ``delta_cached`` more (or
        # fewer) chunks resident in the buffer pool from forward into
        # backward. Each delta cache hit skips one all-gather collective
        # in backward — the paper's "buffers surviving forward are reused
        # in backward if not evicted, skipping reload" invariant. Without
        # this translation the chunked-wall override is FLAT in
        # ``n_buffer`` and the searcher's "argmin over n_buffer" would
        # collapse to the minimum-feasible value (``min_n_buffer_for``);
        # the searcher then picks ``n_buffer=2`` for a Mode-C workload
        # where ``n_buffer >= 6`` would let most non-persistent chunks
        # survive forward and skip the re-gather in backward.
        #
        # The savings-per-delta-hit is the backward NCCL gather PLUS the
        # H2D reload that an uncached chunk would have to pay before the
        # gather. Mirrors
        # ``t_bwd_comm_per_chunk_uncached - t_bwd_comm_per_chunk_cached
        # = collective + S_chunk/<bw>`` in the analytical branch
        # below (post CodeRabbit Round-5 R5-B fix), keeping the two
        # paths' n_buffer-coefficients consistent. Pre-R5-B this term
        # was just ``nccl_gather`` and so under-credited buffer cache
        # hits in the phase-2 override path on PCIe-bound single-rank
        # configs.
        n_nonpersist_bootstrap = max(0, layout.N_chunk - trace.phase2_n_persist)
        bootstrap_cached = min(trace.phase2_n_buffer, n_nonpersist_bootstrap)
        candidate_cached = min(n_buffer, n_nonpersist)
        delta_cached = candidate_cached - bootstrap_cached
        # Savings per cache hit = backward gather collective skipped +
        # H2D reload skipped. Single-rank / no-collective case has
        # nccl_gather=0 (PCIe-only term remains); a pathological bw<=0
        # collapses the H2D term to 0 (matching ``_comm_time_chunk``'s
        # defensive division).
        #
        # Bandwidth choice for the H2D save: under the per-chunk
        # timeline-overlap model (paper §3.3), the marginal cached
        # chunk lives at the high-index END of the non-persistent
        # range — far from the SWAP-early block placement at indices
        # ``[0, n_swap)`` (``layout_rules.assign_modes`` rule 1). Its
        # prefetch window therefore overlaps no SWAP block, so its
        # per-chunk effective bandwidth IS ``full_h2d``. We use
        # ``full_h2d`` here on that basis. (For configs where the
        # cached pool dips into low-index chunks adjacent to SWAP
        # blocks the marginal save would be slightly higher than
        # ``S_chunk / full_h2d`` — those chunks are derated — but the
        # conservative direction is to under-credit, not over-credit,
        # the cache hit. Refining to consult per-chunk bw vectors
        # here is a follow-up if a measured discrepancy emerges.)
        save_h2d_bw = full_h2d
        h2d_save_per_hit = layout.S_chunk / save_h2d_bw if save_h2d_bw > 0 else 0.0
        gather_save_per_hit = nccl_gather + h2d_save_per_hit
        # Net override: subtract delta-hit savings from the measured
        # backward. Clamp at 0 to prevent negative t_bwd if a wildly
        # noisy trace has more savings than measured backward (would
        # only happen on a degenerate bootstrap that already cached
        # everything).
        t_bwd_buffer_correction = -delta_cached * gather_save_per_hit
        t_bwd = max(
            0.0,
            t_bwd_compute_total + t_bwd_swap_prefetch + t_bwd_buffer_correction,
        )
    else:
        if layout.N_chunk > 0:
            t_bwd_compute_per_chunk = t_bwd_compute_total / layout.N_chunk
        else:
            t_bwd_compute_per_chunk = 0.0

        # Buffer-cached vs. uncached split (per-chunk). The last
        # ``n_buffer`` non-persistent chunks (highest indices in
        # backward-order — chunks the buffer pool retains across the
        # forward→backward boundary) skip re-gather; the rest pay the
        # full round-trip. Per-chunk loop fuses the cache-hit predicate
        # with the per-chunk timeline-overlap bandwidth — no separate
        # affected/unaffected layering, no four-way split: each chunk's
        # cost is just ``t_bwd_comm_per_chunk_{cached,uncached}[cid]``
        # (already evaluated at the chunk's per-chunk effective
        # bandwidth above).
        n_cached = min(n_buffer, n_nonpersist)
        # Cached chunks are the LAST ``n_cached`` non-persistent chunks
        # (chunks ``[N_chunk - n_cached, N_chunk)``). This matches the
        # buffer pool's LRU residency invariant: the most-recently-used
        # chunks at the end of forward are the ones still resident
        # when backward starts walking blocks in reverse.
        cached_threshold = layout.N_chunk - n_cached

        # Persistent chunks: paper Eq. 6 first branch — only the
        # reduce-scatter collective contributes to comm (no gather, no
        # H2D, no D2H grad-offload because the chunk lives on GPU).
        # Paper Eq. 5 backward roofline is max(compute, comm) per chunk.
        # Persistent chunks have no PCIe traffic, so the per-chunk
        # contention model does not apply.
        t_bwd_persistent_chunks = n_persist * max(t_bwd_compute_per_chunk, nccl_reduce)
        t_bwd_nonpersistent_chunks = 0.0
        for cid in range(n_persist, layout.N_chunk):
            if cid >= cached_threshold:
                comm = t_bwd_comm_per_chunk_cached[cid]
            else:
                comm = t_bwd_comm_per_chunk_uncached[cid]
            t_bwd_nonpersistent_chunks += max(t_bwd_compute_per_chunk, comm)
        t_bwd = (
            t_bwd_persistent_chunks + t_bwd_nonpersistent_chunks + t_bwd_swap_prefetch
        )

    # ----- Optimizer step ----------------------------------------------
    # Model-state bytes per chunk = model_state_bytes / N_chunk.
    if layout.N_chunk > 0:
        ms_per_chunk = trace.model_state_bytes / layout.N_chunk
    else:
        ms_per_chunk = 0.0

    # ``cpu_adam_bytes_per_sec == 0`` is the sentinel ``measure_cpu_adam``
    # emits when DeepSpeedCPUAdam can't be imported or constructed
    # (e.g. CUDA-version mismatch on this rig). The runtime path mirrors
    # this: ``protrain_optimizer_wrapper`` sets ``cpu_optim = None`` and
    # **skips the CPU step entirely** for non-persistent chunks (they sit
    # un-stepped — a "training-incorrect" state the wrapper LOG.errors
    # about). Earlier this branch fell back to a hardcoded prior, which
    # billed a fictional CPU-Adam wall and made the searcher pick configs
    # that minimized a cost the runtime would never pay. Now we honour
    # the absence: ``cpu_adam_bps = 0.0`` here is a sentinel that drops
    # the ``t_cpu_optim`` term to 0 below.
    if hw.cpu_adam_bytes_per_sec > 0.0:
        cpu_adam_bps = hw.cpu_adam_bytes_per_sec
    else:
        LOG.warning(
            "estimate_runtime: cpu_adam_bytes_per_sec=0 — treating CPU "
            "Adam as unavailable (matches optim_wrapper's cpu_optim=None "
            "path). Non-persistent chunks contribute 0 to t_cpu_optim. "
            "Note that under this state non-persistent chunks are NOT "
            "actually being stepped at runtime either; install/fix "
            "DeepSpeed for full coverage."
        )
        cpu_adam_bps = 0.0  # sentinel — t_cpu_optim collapses to 0

    if hw.gpu_adam_bytes_per_sec > 0.0:
        gpu_adam_bps = hw.gpu_adam_bytes_per_sec
    else:
        LOG.warning(
            "estimate_runtime: gpu_adam_bytes_per_sec unavailable; using "
            "fallback %.2e (re-run profiler for a calibrated rate)",
            _GPU_ADAM_FALLBACK,
        )
        gpu_adam_bps = _GPU_ADAM_FALLBACK

    t_gpu_optim = n_persist * ms_per_chunk / gpu_adam_bps
    # In ZeRO-3/Mode-C, non-persistent chunks are sharded across ranks, so
    # each rank only Adam-steps ``1/world_size`` of every chunk. Without
    # this divide the CPU-optim cost was billed at ``world_size x`` actual
    # — the searcher consequently under-rated configs with high
    # ``n_nonpersist``. Mode-B (DDP-replicated, no sharding) leaves every
    # rank stepping the full chunk, so the divide stays gated on
    # ``zero3_shard``.
    cpu_shard_divisor = max(1, hw.gpu_count) if hw.zero3_shard else 1
    if cpu_adam_bps <= 0.0:
        # CPU Adam unavailable — non-persistent chunks won't actually be
        # stepped at runtime (``optim_wrapper`` sets ``cpu_optim = None``
        # and skips the CPU step, leaving those chunks un-updated — a
        # training-incorrect state the wrapper LOG.errors about).
        # Mark configs that offload chunks as INFEASIBLE so the searcher's
        # argmin doesn't pick them on a fictional ``t_cpu_optim=0`` ranking.
        # Configs with ``n_nonpersist == 0`` (everything persistent on GPU,
        # e.g. small LoRA fits) remain feasible because no CPU step is
        # required at runtime.
        if n_nonpersist > 0:
            return float("inf")
        t_cpu_optim = 0.0
    else:
        t_cpu_optim = n_nonpersist * (ms_per_chunk / cpu_shard_divisor) / cpu_adam_bps

    # TODO(coderabbit-pr10-7b-residual): the phase-2 chunked-wall
    # measurements (``trace.steady_fwd_chunked_wall_s`` /
    # ``steady_bwd_chunked_wall_s``, consumed at lines 545-546 / 590-647)
    # are captured under the bootstrap config (``n_persist=0+pinned``)
    # and consumed as flat baselines independent of candidate
    # ``n_persist``. In single-rank mode the only ``n_persist``-related
    # term (``gather_save_per_hit`` at ~line 636) is gated on
    # ``nccl_gather`` and short-circuits to 0 when ``world_size==1``, so
    # candidates with high ``n_persist`` get the same chunked-wall as the
    # bootstrap's ``n_persist=0`` measurement. On 7B-LoRA this leaves a
    # ~19% over-prediction residual after the cpu_adam_bps fix above.
    # Real fix needs an analytical PCIe-roundtrip translation across
    # ``n_persist`` (or a higher-``n_persist`` re-bootstrap) — multi-day
    # refactor, deferred per the v1 paper-alignment scope policy.

    # Eq. 2: T_iter = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)
    t_iter = t_fwd + max(t_bwd + t_gpu_optim, t_cpu_optim)

    LOG.debug(
        "estimate_runtime: cfg=%s t_fwd=%.4fs t_bwd=%.4fs t_gpu_opt=%.4fs "
        "t_cpu_opt=%.4fs -> t_iter=%.4fs",
        cfg,
        t_fwd,
        t_bwd,
        t_gpu_optim,
        t_cpu_optim,
        t_iter,
    )
    # Silence unused n_block — kept for debug/extension symmetry.
    _ = n_block
    return t_iter


__all__ = ["estimate_runtime"]
