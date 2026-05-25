"""Exhaustive 5-knob search for ProTrain."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost.bandwidth import effective_bw_for_chunk
from axolotl.integrations.protrain.cost.memory import (  # noqa: F401 - re-exported for test back-compat
    estimate_cpu_footprint,
    estimate_peak,
)
from axolotl.integrations.protrain.cost.runtime import estimate_runtime
from axolotl.integrations.protrain.search.knobs import derive_bounds
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    Bounds,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
    SearchResult,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class _BlockMapSkeleton:
    """n_persist-invariant slice of ``_block_map_peak_contribution``.

    ``base_max`` is the max of (per-fwd-op base contribution) excluding OFFLOAD
    bumps; ``offload_candidates`` holds the (op_base_value, chunks_tuple) pairs
    for each OFFLOAD block's last-fwd-op. ``_apply_persistent_adjustments`` then
    only needs to scan offload_candidates and max ``base_max`` against
    ``op_base_value + s_chunk`` for any candidate that isn't fully persistent.
    """

    base_max: int
    offload_candidates: tuple[tuple[int, tuple[int, ...]], ...]
    s_chunk: int
    has_forward: bool
    degenerate_total: int


def min_n_buffer_for(layout: ChunkLayout, n_persist: int) -> int:
    """Minimum n_buffer for the scheduler's lookahead prefetch at this n_persist."""
    persistent: set[ChunkId] = set(layout.effective_persistent_ids(n_persist))
    if len(persistent) >= layout.N_chunk:
        return 0
    block_ids = sorted(layout.block_to_chunks.keys())
    if not block_ids:
        # Sparse layout: pool needs ≥1 slot for any non-persistent chunk.
        return 1
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
    # ≥1 buffer required when any non-persistent chunk exists.
    return max(1, need)


def block_map_runtime_admissible(
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    n_persist: int,
) -> bool:
    """Return True iff the block strategy is safe for current chunk offload.

    CKPT/OFFLOAD/SWAP are always admissible; NONE requires all chunks persistent.
    """
    persistent = set(layout.effective_persistent_ids(n_persist))
    for bid, chunks in layout.block_to_chunks.items():
        mode = block_map.get(bid, BlockMode.NONE)
        if (
            mode is BlockMode.CKPT
            or mode is BlockMode.OFFLOAD
            or mode is BlockMode.SWAP
        ):
            continue
        if any(ChunkId(int(cid)) not in persistent for cid in chunks):
            return False
    return True


def _iter_candidates(bounds: Bounds) -> Iterator[CostConfig]:
    """Enumerate feasible ``CostConfig`` tuples within ``bounds``."""
    n_chunk = bounds.N_chunk
    n_block = bounds.N_block
    n_interval = bounds.N_interval

    for n_ckpt in range(0, n_block + 1):
        for n_offload in range(0, n_block - n_ckpt + 1):
            # n_swap bounded by (a) blocks remaining after
            # ckpt+offload, (b) N_interval.
            max_swap = min(n_block - n_ckpt - n_offload, n_interval)
            for n_swap in range(0, max_swap + 1):
                for n_persist in range(0, n_chunk + 1):
                    # n_buffer fills the remainder of chunk budget.
                    max_buffer = n_chunk - n_persist
                    for n_buffer in range(0, max_buffer + 1):
                        yield CostConfig(
                            n_persist=n_persist,
                            n_buffer=n_buffer,
                            n_swap=n_swap,
                            n_checkpoint=n_ckpt,
                            n_offload=n_offload,
                        )


def _block_map_peak_contribution(
    block_map: BlockStrategyMap,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    *,
    forward_ops_by_block: dict[BlockId, list[int]] | None = None,
    tree_index_map: dict[BlockId, int] | None = None,
    n_persist: int | None = None,
) -> int:
    """Compute the block-map-dependent part of the raw peak (excluding n_persist/n_buffer)."""
    from axolotl.integrations.protrain.cost.memory import (
        block_tree_index_map,
        cross_attn_persist_bytes,
        op_cross_attn_surcharge,
    )

    if forward_ops_by_block is None:
        forward_ops_by_block = defaultdict(list)
        for i, op in enumerate(trace.op_order):
            if op.is_forward and op.block_id is not None:
                forward_ops_by_block[op.block_id].append(i)

    # CKPT bump at first fwd op; OFFLOAD bump at last fwd op (skip when all chunks persistent).
    ckpt_bump_op: dict[int, int] = {}
    offload_bump_op: dict[int, int] = {}
    persistent_chunks: set[ChunkId] | None = None
    if n_persist is not None:
        persistent_chunks = set(layout.effective_persistent_ids(n_persist))
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)
        elif mode is BlockMode.OFFLOAD:
            if persistent_chunks is not None:
                chunks = layout.block_to_chunks.get(block_id, ())
                # All-persistent block: no gather, no bump.
                if not chunks or all(
                    ChunkId(int(cid)) in persistent_chunks for cid in chunks
                ):
                    continue
            offload_bump_op[op_idxs[-1]] = int(block_id)

    # Cumulative NONE/OFFLOAD activation bytes per fwd-op index.
    block_first_op = {bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops}
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])
    cumulative_none: list[tuple[int, int]] = []  # (first_op_idx, cumulative)
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            running += trace.activation_sizes.get(bid, 0)
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    if tree_index_map is None:
        tree_index_map = block_tree_index_map(trace)
    cross_attn_bytes = cross_attn_persist_bytes(trace, block_map, tree_index_map)

    s_chunk = layout.S_chunk
    best = 0
    have_any_forward = False
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        have_any_forward = True
        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)
        ckpt_extra = 0
        if i in ckpt_bump_op:
            ckpt_extra = trace.activation_sizes.get(BlockId(ckpt_bump_op[i]), 0)
        offload_extra = 0
        if i in offload_bump_op:
            offload_extra = s_chunk
        op_cross_attn = op_cross_attn_surcharge(op, cross_attn_bytes, tree_index_map)
        candidate = (
            live_none + ckpt_extra + offload_extra + op_cross_attn + intra + inter
        )
        if candidate > best:
            best = candidate

    if not have_any_forward:
        # Degenerate trace fallback: NONE/OFFLOAD retained activation total.
        total_none = 0
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            mode = block_map.get(bid, BlockMode.NONE)
            if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
                total_none += act_sz
        return total_none

    return best


def _build_block_map_skeleton(
    block_map: BlockStrategyMap,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    *,
    forward_ops_by_block: dict[BlockId, list[int]],
    tree_index_map: dict[BlockId, int],
) -> _BlockMapSkeleton:
    """Compute the n_persist-invariant op-walk skeleton for ``_block_map_peak_contribution``.

    ``base_max`` is the max of per-fwd-op base contributions (live_none +
    ckpt_extra + cross_attn + intra + inter). Each OFFLOAD block contributes
    ``(per_op_base_at_last_fwd_op, chunks_tuple)`` so the per-n_persist pass
    only re-evaluates the bumps. Bit-equivalence with
    ``_block_map_peak_contribution`` follows from OFFLOAD bumps being a
    constant ``s_chunk`` per block at a specific op_idx.
    """
    from axolotl.integrations.protrain.cost.memory import (
        cross_attn_persist_bytes,
        op_cross_attn_surcharge,
    )

    ckpt_bump_op: dict[int, int] = {}
    offload_block_ops: list[tuple[int, tuple[int, ...]]] = []
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)
        elif mode is BlockMode.OFFLOAD:
            chunks = tuple(int(cid) for cid in layout.block_to_chunks.get(block_id, ()))
            offload_block_ops.append((op_idxs[-1], chunks))

    block_first_op = {bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops}
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])
    cumulative_none: list[tuple[int, int]] = []
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            running += trace.activation_sizes.get(bid, 0)
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    cross_attn_bytes = cross_attn_persist_bytes(trace, block_map, tree_index_map)

    per_op_base_by_idx: dict[int, int] = {}
    base_max = 0
    have_any_forward = False
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        have_any_forward = True
        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)
        ckpt_extra = 0
        if i in ckpt_bump_op:
            ckpt_extra = trace.activation_sizes.get(BlockId(ckpt_bump_op[i]), 0)
        op_cross_attn = op_cross_attn_surcharge(op, cross_attn_bytes, tree_index_map)
        v = live_none + ckpt_extra + op_cross_attn + intra + inter
        per_op_base_by_idx[i] = v
        if v > base_max:
            base_max = v

    offload_candidates: list[tuple[int, tuple[int, ...]]] = [
        (per_op_base_by_idx[op_idx], chunks) for op_idx, chunks in offload_block_ops
    ]

    degenerate_total = 0
    if not have_any_forward:
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            mode = block_map.get(bid, BlockMode.NONE)
            if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
                degenerate_total += act_sz

    return _BlockMapSkeleton(
        base_max=base_max,
        offload_candidates=tuple(offload_candidates),
        s_chunk=layout.S_chunk,
        has_forward=have_any_forward,
        degenerate_total=degenerate_total,
    )


def _apply_persistent_adjustments(
    skeleton: _BlockMapSkeleton,
    persistent_chunks: frozenset[ChunkId] | set[ChunkId] | None,
) -> int:
    """Apply OFFLOAD bumps for non-all-persistent blocks and return the peak.

    For each OFFLOAD block whose chunks aren't all persistent, its contribution
    is ``per_op_base_at_last_fwd_op + s_chunk``. The final peak is the max of
    ``base_max`` and those active candidates. When ``persistent_chunks`` is
    ``None`` (legacy callers / n_offload=0 hoist path), every candidate fires.
    """
    if not skeleton.has_forward:
        return skeleton.degenerate_total

    best = skeleton.base_max
    s_chunk = skeleton.s_chunk
    for op_base_val, chunks in skeleton.offload_candidates:
        if (
            persistent_chunks is not None
            and chunks
            and all(ChunkId(int(cid)) in persistent_chunks for cid in chunks)
        ):
            continue
        candidate = op_base_val + s_chunk
        if candidate > best:
            best = candidate
    return best


def search(
    trace: ProfilerTrace,
    layout: ChunkLayout,
    capacity_bytes: int,
    hw: HardwareProfile,
    cpu_capacity_bytes: int | None = None,
    *,
    forbid_activation_offload: bool = False,
    prefer_no_offload_on_non_nvlink: bool = True,
    non_nvlink_multi_rank: bool | None = None,
) -> SearchResult:
    """Return the minimum-runtime SearchResult fitting under capacity_bytes.

    ``forbid_activation_offload``: skip any candidate with ``n_offload > 0``.
    Wired from ``cfg.lora_mlp_kernel`` — the fused MLP backward kernel returns
    a real gradient on offloaded activations whose ChunkManager placeholder is
    zero-shape, crashing at the first backward with a shape-mismatch error.

    ``prefer_no_offload_on_non_nvlink``: defensive tie-break that prefers
    ``n_offload=0`` configs when the rig is detected as non-NVLink multi-rank.
    v71/v72-redux verified bs=2 with ``n_offload > 0`` hangs in autograd
    backward on consumer multi-GPU rigs; v62-style (``n_persist=128,
    n_offload=0``) runs cleanly. Configs within a 5% noise band of the best
    predicted runtime are re-ranked by (n_offload ASC, n_swap ASC,
    -n_persist ASC) before the existing (n_ckpt, -n_persist, n_buffer)
    tie-break runs. ``non_nvlink_multi_rank``: explicit override; when None
    the heuristic auto-detects via ``hw.gpu_count > 1 and not hw.has_nvlink``.
    """
    bounds = derive_bounds(trace, layout)

    _ = hw.zero3_shard  # noqa: F841

    # Detect non-NVLink multi-rank: gpu_count>1 AND no NVLink topology. The
    # hardware.py default leaves has_nvlink=False; on real NVLink rigs the
    # caller is expected to pass non_nvlink_multi_rank=False to disable.
    if non_nvlink_multi_rank is None:
        non_nvlink_multi_rank = bool(hw.gpu_count > 1 and not hw.has_nvlink)
    apply_no_offload_heuristic = bool(
        prefer_no_offload_on_non_nvlink and non_nvlink_multi_rank
    )
    _NON_NVLINK_TIE_RATIO = 0.05  # 5% noise band for the n_offload tie-break

    n_total = 0
    n_feasible = 0
    n_gpu_feasible = 0  # cleared GPU gate (used to disambiguate failure mode)
    n_cpu_rejected = 0  # cleared GPU gate but failed CPU gate
    # cleared GPU+CPU gates but estimate_runtime returned non-finite
    n_runtime_rejected = 0
    n_kernel_filtered = 0  # n_offload>0 skipped under forbid_activation_offload
    best_iter_s: float = float("inf")
    best_cfg: CostConfig | None = None
    best_block_map: BlockStrategyMap | None = None
    best_peak: int = 0
    # Track closest-to-feasible config for actionable error messages.
    closest_cfg: CostConfig | None = None
    closest_peak: int = 0
    closest_deficit: int = 0

    # Pre-compute F(block_map) once per (n_swap, n_ckpt, n_offload).
    from axolotl.integrations.protrain.cost.memory import (
        ALPHA_FRAGMENTATION,  # noqa: F401
        alpha_fragmentation_for_cfg,
        apply_hot_iter_cap,
        block_tree_index_map,
        hot_iter_peak_cap,
        model_state_present_bytes,
    )

    s_chunk = layout.S_chunk

    # Hoist trace-only maps; depend on trace only, not block_map.
    forward_ops_by_block: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if op.is_forward and op.block_id is not None:
            forward_ops_by_block[op.block_id].append(i)
    tree_index_map = block_tree_index_map(trace)

    for n_ckpt in range(0, bounds.N_block + 1):
        # Mode-aware fragmentation alpha: 4-bit + n_ckpt>0 uses the
        # Mode-C-CKPT calibration (0.95) instead of the Mode-A 0.75.
        # Other dtypes ignore the n_ckpt split.
        alpha = alpha_fragmentation_for_cfg(
            hw.dominant_param_bytes_per_element,
            CostConfig(
                n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=n_ckpt, n_offload=0
            ),
        )
        for n_offload in range(0, bounds.N_block - n_ckpt + 1):
            if forbid_activation_offload and n_offload > 0:
                # Drop the n_offload>0 subtree wholesale; the fused MLP kernel
                # is incompatible with chunk-storage placeholders.
                n_kernel_filtered += 1
                continue
            max_swap = min(bounds.N_block - n_ckpt - n_offload, bounds.N_interval)
            for n_swap in range(0, max_swap + 1):
                block_map = assign_modes(
                    n_swap, n_ckpt, bounds.N_block, n_offload=n_offload
                )

                # Inner loop maximises n_buffer within capacity; F_bm uses uncapped raw-peak.
                # Probe with n_persist=N_chunk so the cap-domination check honours full Adam state.
                _cap_probe_cfg = CostConfig(
                    n_persist=bounds.N_chunk,
                    n_buffer=0,
                    n_swap=n_swap,
                    n_checkpoint=n_ckpt,
                    n_offload=n_offload,
                )
                _hot_cap = hot_iter_peak_cap(
                    trace, block_map, _cap_probe_cfg, layout=layout
                )
                if _hot_cap is not None:
                    _probe_model_state = model_state_present_bytes(
                        _cap_probe_cfg, layout, trace
                    )
                    _probe_raw = apply_hot_iter_cap(
                        _probe_model_state + _hot_cap,
                        _probe_model_state,
                        _hot_cap,
                        layout,
                    )
                    _cap_dominates = int(alpha * _probe_raw) <= capacity_bytes
                else:
                    _cap_dominates = False

                # Build the n_persist-invariant skeleton once per (n_swap, n_ckpt, n_offload).
                # n_offload==0 → no OFFLOAD candidates → skeleton.base_max is the full peak.
                # n_offload>0 → apply per-n_persist OFFLOAD bumps via the skeleton.
                _bm_skeleton = _build_block_map_skeleton(
                    block_map,
                    trace,
                    layout,
                    forward_ops_by_block=forward_ops_by_block,
                    tree_index_map=tree_index_map,
                )
                f_bm_invariant: int | None
                if n_offload == 0:
                    f_bm_invariant = _apply_persistent_adjustments(
                        _bm_skeleton, persistent_chunks=None
                    )
                else:
                    f_bm_invariant = None

                # Hoist effective_bw_for_chunk: depends on (chunk_id, cfg.n_swap,
                # layout, block_map, hw) — all loop-invariant within the inner
                # n_persist/n_buffer loop. Build once, pass through estimate_runtime.
                _bw_probe_cfg = CostConfig(
                    n_persist=0,
                    n_buffer=0,
                    n_swap=n_swap,
                    n_checkpoint=n_ckpt,
                    n_offload=n_offload,
                )
                _chunk_bw_table: dict[
                    int, tuple[tuple[float, float], tuple[float, float]]
                ] = {}
                for _cid in range(bounds.N_chunk):
                    _bw_fwd = effective_bw_for_chunk(
                        ChunkId(_cid),
                        _bw_probe_cfg,
                        hw,
                        layout,
                        block_map,
                        direction="fwd",
                    )
                    _bw_bwd = effective_bw_for_chunk(
                        ChunkId(_cid),
                        _bw_probe_cfg,
                        hw,
                        layout,
                        block_map,
                        direction="bwd",
                    )
                    _chunk_bw_table[_cid] = (_bw_fwd, _bw_bwd)

                # Partition: |prefix U mandatory_persistent| + n_buffer <= N_chunk.
                for n_persist in range(0, bounds.N_chunk + 1):
                    if f_bm_invariant is not None:
                        f_bm = f_bm_invariant
                    else:
                        f_bm = _apply_persistent_adjustments(
                            _bm_skeleton,
                            persistent_chunks=layout.effective_persistent_ids(
                                n_persist
                            ),
                        )
                    if _cap_dominates:
                        max_sum = bounds.N_chunk
                    elif alpha > 0 and s_chunk > 0:
                        max_sum = int((capacity_bytes / alpha - f_bm) / s_chunk)
                    else:
                        max_sum = bounds.N_chunk
                    max_sum = max(0, min(max_sum, bounds.N_chunk))

                    # Max n_buffer: partition bound uses augmented persistent count (prefix U mandatory).
                    persistent_count_aug = len(
                        layout.effective_persistent_ids(n_persist)
                    )
                    max_buffer = min(
                        bounds.N_chunk - persistent_count_aug,
                        max_sum - persistent_count_aug,
                    )
                    if max_buffer < 0:
                        # OFFLOAD: budget may re-open at higher n_persist; only break on monotone n_offload==0.
                        if f_bm_invariant is not None:
                            break
                        continue

                    # Scheduler lookahead needs enough buffers for current + next block's non-persistent chunks.
                    min_buffer = min_n_buffer_for(layout, n_persist)
                    if min_buffer > max_buffer:
                        continue
                    if not block_map_runtime_admissible(layout, block_map, n_persist):
                        continue

                    # 2-point shortcut (min, max): runtime is monotone-decreasing in
                    # n_buffer (more cache hits = lower comm time) and CPU footprint
                    # is invariant in n_buffer (chunk_term + swap_term depend on
                    # n_persist / n_swap only — see ``estimate_cpu_footprint``). So
                    # interior values of n_buffer are dominated by max_buffer on
                    # runtime and identical to it on the CPU gate; min_buffer is kept
                    # for the noise-band tie-break (lower n_buffer preferred). The
                    # CPU-gate branch previously enumerated the full range, blowing
                    # search time up ~50x at N_chunk=130 (v67 measurement: 9.5min
                    # init for Mode B s1).
                    n_buffer_candidates: Iterable[int]
                    if min_buffer == max_buffer:
                        n_buffer_candidates = (min_buffer,)
                    else:
                        n_buffer_candidates = (min_buffer, max_buffer)
                    for n_buffer in n_buffer_candidates:
                        n_total += 1
                        cfg = CostConfig(
                            n_persist=n_persist,
                            n_buffer=n_buffer,
                            n_swap=n_swap,
                            n_checkpoint=n_ckpt,
                            n_offload=n_offload,
                        )
                        # Single-source model_state via model_state_present_bytes to match estimate_peak.
                        model_state_present = model_state_present_bytes(
                            cfg, layout, trace
                        )
                        raw_peak = model_state_present + f_bm
                        # Reuse outer-loop _hot_cap: hot_iter_peak_cap depends only on
                        # (trace, block_map, layout) in the steady_fwd_block_peak path,
                        # and on (n_ckpt, n_swap, n_offload) — all inner-loop-invariant —
                        # in the aggregate fallback. Calling per-candidate burned ~18s
                        # for 173k iterations at bs=2 (v69 profile).
                        raw_peak = apply_hot_iter_cap(
                            raw_peak, model_state_present, _hot_cap, layout
                        )
                        predicted_peak = int(alpha * raw_peak) if raw_peak > 0 else 0
                        if predicted_peak > capacity_bytes:
                            deficit = predicted_peak - capacity_bytes
                            if closest_cfg is None or deficit < closest_deficit:
                                closest_cfg = cfg
                                closest_peak = predicted_peak
                                closest_deficit = deficit
                            continue
                        n_gpu_feasible += 1
                        # Optional CPU-RAM gate (per-rank pinned bytes).
                        if cpu_capacity_bytes is not None:
                            cpu_footprint = estimate_cpu_footprint(
                                cfg, layout, hw, trace=trace
                            )
                            if cpu_footprint > cpu_capacity_bytes:
                                n_cpu_rejected += 1
                                continue
                        n_feasible += 1
                        predicted_iter_s = estimate_runtime(
                            cfg,
                            trace,
                            layout,
                            block_map,
                            hw,
                            chunk_bw_table=_chunk_bw_table,
                        )
                        # Non-finite runtime: track separately to disambiguate failure mode.
                        if not math.isfinite(predicted_iter_s):
                            n_runtime_rejected += 1
                            continue
                        # Near-tie aware comparator (1% noise floor); prefer (lower ckpt, higher persist, lower buffer).
                        _NEAR_TIE_RATIO = 0.01
                        if best_cfg is None:
                            best_iter_s = predicted_iter_s
                            best_cfg = cfg
                            best_block_map = block_map
                            best_peak = predicted_peak
                        else:
                            improvement = best_iter_s - predicted_iter_s
                            # PR #17c: on non-NVLink multi-rank rigs, prefer n_offload=0
                            # within a 5% noise band of the current best (covers both
                            # slightly-faster and slightly-slower candidates). v71/v72-
                            # redux: bs=2 + n_offload>0 hangs in autograd backward on
                            # consumer multi-GPU; v62-style n_persist=128 / n_offload=0
                            # runs cleanly. Bounded by _NON_NVLINK_TIE_RATIO; outside
                            # the band the original 1%-noise comparator wins.
                            non_nvlink_swap = False
                            if (
                                apply_no_offload_heuristic
                                and cfg.n_offload != best_cfg.n_offload
                                and abs(improvement)
                                <= best_iter_s * _NON_NVLINK_TIE_RATIO
                            ):
                                cur_nv_key = (
                                    cfg.n_offload,
                                    cfg.n_swap,
                                    -cfg.n_persist,
                                )
                                best_nv_key = (
                                    best_cfg.n_offload,
                                    best_cfg.n_swap,
                                    -best_cfg.n_persist,
                                )
                                if cur_nv_key < best_nv_key:
                                    non_nvlink_swap = True
                            if non_nvlink_swap:
                                best_iter_s = predicted_iter_s
                                best_cfg = cfg
                                best_block_map = block_map
                                best_peak = predicted_peak
                            elif improvement >= best_iter_s * _NEAR_TIE_RATIO:
                                best_iter_s = predicted_iter_s
                                best_cfg = cfg
                                best_block_map = block_map
                                best_peak = predicted_peak
                            elif improvement > 0:
                                # In noise band: tie-break by (n_ckpt, -n_persist, n_buffer).
                                cur_key = (
                                    cfg.n_checkpoint,
                                    -cfg.n_persist,
                                    cfg.n_buffer,
                                )
                                best_key = (
                                    best_cfg.n_checkpoint,
                                    -best_cfg.n_persist,
                                    best_cfg.n_buffer,
                                )
                                if cur_key < best_key:
                                    best_iter_s = predicted_iter_s
                                    best_cfg = cfg
                                    best_block_map = block_map
                                    best_peak = predicted_peak

    if best_cfg is None or best_block_map is None:
        # Disambiguate runtime-rejection vs capacity-rejection failure modes.
        if n_feasible > 0 and n_runtime_rejected == n_feasible:
            raise RuntimeError(
                "no ProTrain config has a finite runtime estimate; every "
                f"capacity-feasible config (out of {n_feasible}) was "
                "rejected by estimate_runtime (likely CPU-Adam unavailable "
                "for non-persistent chunks on this setup). Evaluated "
                f"{n_total} configs total."
            )
        # CPU-bound failure: GPU gate cleared but CPU envelope exceeded.
        if (
            cpu_capacity_bytes is not None
            and n_gpu_feasible > 0
            and n_cpu_rejected == n_gpu_feasible
        ):
            raise RuntimeError(
                f"no ProTrain config fits in {cpu_capacity_bytes / 1e9:.1f} GB "
                f"host RAM (per-rank CPU budget); {n_gpu_feasible} configs "
                f"cleared the GPU capacity gate but all exceeded the CPU "
                f"footprint limit. Evaluated {n_total} configs total. "
                "Scale up: more nodes, more system RAM, or a smaller model."
            )
        # Kernel-feasibility failure: every n_offload=0 candidate is over budget.
        if forbid_activation_offload and n_kernel_filtered > 0:
            raise RuntimeError(
                "no feasible ProTrain config: ``lora_mlp_kernel: true`` forbids "
                "activation-offload (n_offload>0) because the fused MLP "
                "backward kernel returns a real gradient on offloaded "
                "activations whose chunk placeholder is zero-shape "
                "(LoRA_MLPBackward shape-mismatch crash; see v61 finding). "
                f"The searcher dropped {n_kernel_filtered} n_offload>0 "
                "subtrees but none of the remaining n_offload=0 candidates fit "
                f"under capacity_bytes={capacity_bytes}. Either (a) set "
                "``lora_mlp_kernel: false`` (slower per-step but compatible "
                "with activation-offload), or (b) free up GPU memory "
                "(smaller batch, more cards, smaller model)."
            )
        if closest_cfg is not None:
            closest_str = (
                f"  Closest-to-feasible: {closest_cfg} predicted_peak="
                f"{closest_peak / 1e9:.2f} GB "
                f"(deficit {closest_deficit / 1e9:.2f} GB over the "
                f"{capacity_bytes / 1e9:.2f} GB budget)."
            )
        else:
            closest_str = (
                "  (no infeasible candidates evaluated; capacity budget "
                "may be too tight for any layout at all)"
            )
        raise RuntimeError(
            "no feasible ProTrain config under capacity_bytes="
            f"{capacity_bytes} ({capacity_bytes / 1e9:.2f} GB); "
            f"evaluated {n_total} configs.\n"
            f"{closest_str}\n"
            "Try one of:\n"
            "  - reduce micro_batch_size (largest activation-memory lever)\n"
            "  - reduce sequence_len (linear in activation memory)\n"
            "  - set lora_mlp_kernel: false to unlock activation-offload\n"
            "    (n_offload>0) candidates the kernel-aware searcher rules out\n"
            "  - enable activation offload by leaving the lora_mlp_kernel\n"
            "    guard off so the searcher can pick n_offload>0\n"
            "  - increase per-rank GPU memory: add cards, switch to a larger\n"
            "    SKU, or shard the model across more ranks (raises N_chunk\n"
            "    headroom and lowers per-rank model_state)"
        )

    if cpu_capacity_bytes is not None:
        LOG.info(
            "ProTrain search: evaluated %d configs, %d cleared GPU gate, "
            "%d rejected by CPU gate, %d feasible, picked %s "
            "predicted=%dMB %.3fs (cpu_budget=%.1f GB)",
            n_total,
            n_gpu_feasible,
            n_cpu_rejected,
            n_feasible,
            best_cfg,
            best_peak // (1 << 20),
            best_iter_s,
            cpu_capacity_bytes / 1e9,
        )
    else:
        LOG.info(
            "ProTrain search: evaluated %d configs, %d feasible, picked %s "
            "predicted=%dMB %.3fs",
            n_total,
            n_feasible,
            best_cfg,
            best_peak // (1 << 20),
            best_iter_s,
        )
    if forbid_activation_offload and n_kernel_filtered > 0:
        LOG.info(
            "ProTrain search: lora_mlp_kernel guard skipped %d n_offload>0 "
            "subtrees; picked n_offload=%d.",
            n_kernel_filtered,
            best_cfg.n_offload,
        )
    return SearchResult(
        cfg=best_cfg,
        block_map=best_block_map,
        predicted_peak_bytes=best_peak,
        predicted_iter_s=best_iter_s,
    )


__all__ = [
    "block_map_runtime_admissible",
    "min_n_buffer_for",
    "search",
]
