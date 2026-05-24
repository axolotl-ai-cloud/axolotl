"""Peak-memory reconstruction for the ProTrain searcher."""

from __future__ import annotations

from collections import defaultdict

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    OpRecord,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


_STALE_TRACE_WARNING_EMITTED = False


def _saved_tensor_bytes_per_block(trace: ProfilerTrace) -> dict[BlockId, int]:
    """Per-block saved-for-backward bytes proxy from steady-state forward peaks."""
    deltas: dict[BlockId, int] = {}
    persisted_per_block_peak = getattr(trace, "steady_fwd_block_peak_bytes", None) or {}
    # Normalize key types so JSON/pickle round-trips still hit per_block_peak.get(BlockId).
    per_block_peak: dict[BlockId, int] = {
        BlockId(int(block_id)): int(peak_bytes)
        for block_id, peak_bytes in persisted_per_block_peak.items()
    }
    activation_sizes = trace.activation_sizes or {}
    if not per_block_peak:
        return {BlockId(int(bid)): int(sz) for bid, sz in activation_sizes.items()}

    # Sort by block id to walk in forward order. Keys are already
    # canonical ``BlockId`` after the normalization above, so sorted()
    # operates on int-equivalent NewType values without further coercion.
    sorted_bids = sorted(per_block_peak.keys())
    if not sorted_bids:
        return {BlockId(int(bid)): int(sz) for bid, sz in activation_sizes.items()}

    forward_diffs: list[int] = []
    for prev_bid, cur_bid in zip(sorted_bids, sorted_bids[1:], strict=False):
        prev_peak = per_block_peak.get(prev_bid, 0)
        cur_peak = per_block_peak.get(cur_bid, 0)
        diff = cur_peak - prev_peak
        if diff > 0:
            forward_diffs.append(diff)
            # Forward difference attributes the cumulative-allocated rise
            # between prev and cur to the bytes prev_bid deposited.
            deltas[prev_bid] = diff

    # Last block has no successor to diff against. Use the median of the
    # observed forward diffs as a robust fallback. When ``forward_diffs``
    # is empty (single block, or every diff was non-positive — unusual
    # but possible if profiling captured a non-uniform steady state),
    # fall back to ``activation_sizes`` for every block.
    if forward_diffs:
        import statistics

        median_diff = int(statistics.median(forward_diffs))
        last_bid = sorted_bids[-1]
        if last_bid not in deltas:
            deltas[last_bid] = median_diff

    # Fill gaps from activation_sizes for blocks missing or zero in the per-block peaks.
    for bid_raw, act_sz in activation_sizes.items():
        bid = BlockId(int(bid_raw))
        if bid not in deltas:
            deltas[bid] = int(act_sz)

    return deltas


# Eq. 11 fragmentation factor; fp16/bf16/8-bit default. Per-dtype override in alpha_fragmentation_for_dtype.
ALPHA_FRAGMENTATION: float = 1.10

# bnb-4-bit alpha; calibrated against Mode-A peaks where frozen 4-bit weights
# dominate per-rank residency and the activation slice is small (empirical
# alpha_measured ~0.70 with cushion).
ALPHA_FRAGMENTATION_4BIT: float = 0.75
ALPHA_FRAGMENTATION_4BIT_MODE_A: float = ALPHA_FRAGMENTATION_4BIT

# Mode-C with gradient checkpointing reverses the dominance: activation churn
# is the dominant moving term, and the 0.75 Mode-A factor structurally
# under-predicts the raw peak (audit table Decision 1 shows alpha_steady
# = 1.43 / 1.25 / 1.08 at seq=512/1024/2048 on 30B-Llama Mode-C, i.e. raw
# predictor low by 8-31% at low seq). 0.95 narrows the gap without
# changing the wrapper-side calibration safety semantics.
ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT: float = 0.95


def alpha_fragmentation_for_dtype(bytes_per_element: float) -> float:
    """Return ALPHA_FRAGMENTATION_4BIT for sub-byte dtypes, else ALPHA_FRAGMENTATION.

    Legacy dtype-only entry point; preserved so call sites that lack a
    ``CostConfig`` (e.g. ``predict_init_transient_peak_bytes``) keep their
    Mode-A calibration. Mode-aware call sites should use
    :func:`alpha_fragmentation_for_cfg`.
    """
    if bytes_per_element < 1.0:
        return ALPHA_FRAGMENTATION_4BIT_MODE_A
    return ALPHA_FRAGMENTATION


def alpha_fragmentation_for_cfg(
    bytes_per_element: float, cfg: CostConfig | None
) -> float:
    """Pick the fragmentation factor for ``(dtype, mode)``.

    Mode-A and Mode-C-without-CKPT keep the 0.75 factor (activation slice
    is small, frozen-weight residency dominates). Mode-C-with-CKPT uses
    0.95 because activation churn fragments differently. Non-4-bit dtypes
    keep the unconditional ``ALPHA_FRAGMENTATION`` (1.10) regardless of
    mode.
    """
    if bytes_per_element >= 1.0:
        return ALPHA_FRAGMENTATION
    is_ckpt = bool(cfg is not None and int(getattr(cfg, "n_checkpoint", 0)) > 0)
    if is_ckpt:
        return ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT
    return ALPHA_FRAGMENTATION_4BIT_MODE_A


def _compute_ckpt_chain_bytes(
    trace: ProfilerTrace, block_map: BlockStrategyMap | None
) -> int:
    """Sum activation bytes across CKPT blocks (chain semantics, not max).

    CKPT blocks retain the block-input boundary tensor across the full
    backward window, so the chain is a sum over CKPT blocks rather than a
    single-block max. Used by both :func:`estimate_peak` and the wrapper-side
    ``_reconstruct_f_bm`` so the analytical and calibrated paths agree.
    """
    if not block_map or not trace.activation_sizes:
        return 0
    total = 0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        if block_map.get(bid, BlockMode.NONE) is BlockMode.CKPT:
            total += int(act_sz)
    return total


def _group_ops_by_block(trace: ProfilerTrace) -> dict[BlockId, list[int]]:
    """Return ``{block_id -> [op_positions]}`` for forward ops only."""
    grouped: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        if op.block_id is None:
            continue
        grouped[op.block_id].append(i)
    return grouped


def _tree_index_for_path(module_path: str) -> int:
    """Best-effort tree-index inference from a module path."""
    normalized = module_path.removeprefix("base_model.model.").removeprefix(
        "base_model."
    )
    if normalized.startswith("encoder.") or normalized == "encoder":
        return 0
    if normalized.startswith("decoder.") or normalized == "decoder":
        return 1
    return 0


def block_tree_index_map(
    trace: ProfilerTrace,
) -> dict[BlockId, int]:
    """Map each ``BlockId`` to its forward-order tree index."""
    persisted = getattr(trace, "block_tree_index", None)
    if persisted:
        # Normalise JSON/pickle string keys back to BlockId.
        return {
            BlockId(int(block_id)): int(tree_index)
            for block_id, tree_index in persisted.items()
        }
    seen: dict[BlockId, int] = {}
    for op in trace.op_order:
        if not op.is_forward or op.block_id is None:
            continue
        if op.block_id in seen:
            continue
        seen[op.block_id] = _tree_index_for_path(op.module_path)
    return seen


def _has_multiple_trees(tree_index_map: dict[BlockId, int]) -> bool:
    """Return True iff at least two distinct tree indices are present."""
    if not tree_index_map:
        return False
    indices = set(tree_index_map.values())
    return len(indices) >= 2


def cross_attn_persist_bytes(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Estimate cross-attention saved-state bytes that span trees."""
    if not _has_multiple_trees(tree_index_map):
        return 0
    encoder_bids = sorted(bid for bid, idx in tree_index_map.items() if idx == 0)
    if not encoder_bids:
        return 0
    last_enc_bid = encoder_bids[-1]
    last_enc_mode = block_map.get(last_enc_bid, BlockMode.NONE)
    if last_enc_mode is BlockMode.NONE or last_enc_mode is BlockMode.OFFLOAD:
        # Already counted in retained_none_bytes.
        return 0
    if last_enc_mode is BlockMode.CKPT:
        # ckpt_chain_bytes already covers residual.
        return 0
    return int(trace.activation_sizes.get(last_enc_bid, 0))


def cross_attn_handoff_bytes(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Return encoder-decoder handoff bytes regardless of encoder-last mode (cap-path use)."""
    if not _has_multiple_trees(tree_index_map):
        return 0
    encoder_bids = sorted(bid for bid, idx in tree_index_map.items() if idx == 0)
    if not encoder_bids:
        return 0
    last_enc_bid = encoder_bids[-1]
    last_enc_mode = block_map.get(last_enc_bid, BlockMode.NONE)
    # NONE/OFFLOAD already retain the full block bytes on GPU so the cap need not preserve them again.
    if last_enc_mode is BlockMode.NONE or last_enc_mode is BlockMode.OFFLOAD:
        return 0
    return int(trace.activation_sizes.get(last_enc_bid, 0))


def op_cross_attn_surcharge(
    op: OpRecord,
    cross_attn_bytes: int,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Per-op cross-attention surcharge during decoder forward."""
    if cross_attn_bytes <= 0 or op.block_id is None:
        return 0
    if tree_index_map.get(op.block_id, 0) > 0:
        return cross_attn_bytes
    return 0


def hot_iter_peak_cap(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap,
    cfg: CostConfig | None = None,
    layout: ChunkLayout | None = None,
) -> int | None:
    """Measured ground-truth upper bound on the raw op-walk peak, or None."""
    if trace.steady_fwd_block_peak_bytes:
        forward_max_block_peak = max(trace.steady_fwd_block_peak_bytes.values())
        # Backward-aware: bwd holds grads + saved acts simultaneously; take max with bwd peaks.
        bwd_per_block = getattr(trace, "steady_bwd_block_peak_bytes", None) or {}
        if bwd_per_block:
            backward_max_block_peak = max(bwd_per_block.values())
            forward_max_block_peak = max(
                forward_max_block_peak, backward_max_block_peak
            )
        ckpt_recomp_bump = 0
        has_offload = False
        # Subtract CKPT/SWAP savings from the all-NONE ceiling so the cap shrinks with n_ckpt/n_swap.
        # Use per-block forward-peak deltas (saved-tensor proxy) not activation_sizes (output-only).
        saved_bytes_proxy = _saved_tensor_bytes_per_block(trace)
        # Encoder→decoder handoff: preserve cross-attn bytes when encoder-last is CKPT/SWAP.
        tree_index_map = block_tree_index_map(trace)
        cross_attn_bytes_for_cap = cross_attn_handoff_bytes(
            trace, block_map, tree_index_map
        )
        encoder_last_bid: BlockId | None = None
        if cross_attn_bytes_for_cap > 0:
            encoder_bids = sorted(
                bid for bid, idx in tree_index_map.items() if idx == 0
            )
            if encoder_bids:
                encoder_last_bid = encoder_bids[-1]
        ckpt_swap_savings = 0
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            mode = block_map.get(bid, BlockMode.NONE)
            block_saved = int(saved_bytes_proxy.get(bid, act_sz))
            if (
                encoder_last_bid is not None
                and bid == encoder_last_bid
                and mode in (BlockMode.CKPT, BlockMode.SWAP)
            ):
                # Cross-attn output cannot be reclaimed on encoder-last under CKPT/SWAP.
                block_saved = max(0, block_saved - cross_attn_bytes_for_cap)
            if mode is BlockMode.CKPT:
                if act_sz > ckpt_recomp_bump:
                    ckpt_recomp_bump = act_sz
                ckpt_swap_savings += block_saved
            elif mode is BlockMode.SWAP:
                ckpt_swap_savings += block_saved
            elif mode is BlockMode.OFFLOAD:
                has_offload = True
        offload_bump = layout.S_chunk if (has_offload and layout is not None) else 0
        # Floor activation portion at zero; model-state floor preserved via add-back below.
        profile_time_model_state = (
            layout.N_chunk * layout.S_chunk if layout is not None else 0
        )
        # All-NONE ceiling = model_state + activation; only activation shrinks with CKPT/SWAP.
        all_none_activation_ceiling = max(
            0, forward_max_block_peak - profile_time_model_state
        )
        capped_savings = min(ckpt_swap_savings, all_none_activation_ceiling)
        return (
            profile_time_model_state
            + (all_none_activation_ceiling - capped_savings)
            + ckpt_recomp_bump
            + offload_bump
        )
    # Aggregate fallback; take max of fwd/bwd aggregates since bwd often exceeds fwd.
    bwd_aggregate = int(getattr(trace, "steady_bwd_peak_bytes", 0))
    aggregate_cap = max(int(trace.steady_fwd_peak_bytes), bwd_aggregate)
    if (
        aggregate_cap > 0
        and cfg is not None
        and cfg.n_checkpoint == 0
        and cfg.n_swap == 0
        and cfg.n_offload == 0
    ):
        return aggregate_cap
    return None


def apply_hot_iter_cap(
    raw_peak: int,
    model_state_present: int,
    measured_cap: int | None,
    layout: ChunkLayout,
) -> int:
    """Apply :func:`hot_iter_peak_cap` to ``raw_peak`` using layered decomposition."""
    if measured_cap is None:
        return raw_peak
    # Layered cap on activation portion only; model_state preserved.
    op_walk_portion = max(0, raw_peak - model_state_present)
    # Subtract profile-time params to recover activation ceiling; clamp at 0 for synthetic traces.
    profile_time_model_state = layout.N_chunk * layout.S_chunk
    measured_activation_cap = max(0, measured_cap - profile_time_model_state)
    if op_walk_portion > measured_activation_cap:
        op_walk_portion = measured_activation_cap
    # Reassemble: model_state_present is preserved through the cap.
    return model_state_present + op_walk_portion


# Pool sizing knobs mirrored from block.swap_pool.ActivationSwapPool; keep in sync.
SWAP_PREFETCH_DEPTH: int = 2
SWAP_SLOTS_PER_BLOCK: int = 8


def estimate_cpu_footprint(
    cfg: CostConfig,
    layout: ChunkLayout,
    hw: HardwareProfile,
    trace: ProfilerTrace | None = None,
) -> int:
    """Per-rank pinned CPU bytes held by non-persistent chunks + SWAP slots."""
    n_persist_eff = len(layout.effective_persistent_ids(cfg.n_persist))
    non_persist = max(0, layout.N_chunk - n_persist_eff)
    # trace.world is the real world_size; hw.gpu_count is local-only.
    if hw.zero3_shard:
        per_rank_divisor = trace.world if trace is not None else hw.gpu_count
    else:
        per_rank_divisor = 1
    per_rank_divisor = max(1, per_rank_divisor)
    per_chunk_sharded = (layout.S_chunk + per_rank_divisor - 1) // per_rank_divisor
    chunk_term = non_persist * per_chunk_sharded

    # Swap term rank-local; size slots to per-block aggregate (strict upper bound).
    swap_term = 0
    if cfg.n_swap > 0 and trace is not None and trace.activation_sizes:
        # Normalise stringified keys from JSON/pickle round-trips for correct sort.
        normalized_activation_sizes: dict[BlockId, int] = {
            BlockId(int(bid)): int(sz) for bid, sz in trace.activation_sizes.items()
        }
        # Swap-early rule: first n_swap blocks in BlockId order.
        sorted_bids = sorted(normalized_activation_sizes.keys())
        swap_band = sorted_bids[: cfg.n_swap]
        if swap_band:
            per_block_activation_bytes = max(
                normalized_activation_sizes.get(bid, 0) for bid in swap_band
            )
            slot_bytes = max(1, int(per_block_activation_bytes))
            swap_term = (
                cfg.n_swap * SWAP_SLOTS_PER_BLOCK * SWAP_PREFETCH_DEPTH * slot_bytes
            )

    return chunk_term + swap_term


def model_state_present_bytes(
    cfg: CostConfig,
    layout: ChunkLayout,
    trace: ProfilerTrace,
) -> int:
    """Resident model-state bytes for the runtime's persistent + buffer set."""
    persistent_ids = layout.effective_persistent_ids(cfg.n_persist)
    n_persist_eff = len(persistent_ids)
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist_eff))

    fp16_total_bytes = layout.N_chunk * layout.S_chunk
    model_state_total = int(getattr(trace, "model_state_bytes", 0) or 0)
    if fp16_total_bytes > 0 and model_state_total > 0:
        # Clamp >= 1.0: aggregate includes fp16 params themselves.
        persistent_factor = max(1.0, model_state_total / fp16_total_bytes)
    else:
        # Process-scoped warn-once: search loop calls this O(candidates) times.
        global _STALE_TRACE_WARNING_EMITTED
        if not _STALE_TRACE_WARNING_EMITTED:
            LOG.warning(
                "model_state_present_bytes: trace.model_state_bytes is missing "
                "or zero (model_state_bytes=%d, fp16_total=%dB); falling back "
                "to the legacy n_persist*S_chunk multiplier. The peak estimate "
                "will UNDER-count full optimizer state — refresh the profiler "
                "trace cache (TRACE_VERSION bump) to restore Eq. 11 fidelity.",
                model_state_total,
                fp16_total_bytes,
            )
            _STALE_TRACE_WARNING_EMITTED = True
        persistent_factor = 1.0
    # Buffer slot during backward: fp16 params + fp16 grads.
    buffer_factor = 2.0
    return int(
        n_persist_eff * layout.S_chunk * persistent_factor
        + n_buffer * layout.S_chunk * buffer_factor
    )


def estimate_peak(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,
) -> int:
    """Estimate steady-state peak GPU memory in bytes."""
    # Delegated so searcher inline fast-path and this validator share Eq. 11 accounting.
    model_state_present = model_state_present_bytes(cfg, layout, trace)

    forward_ops_by_block = _group_ops_by_block(trace)
    tree_index_map = block_tree_index_map(trace)
    cross_attn_bytes = cross_attn_persist_bytes(trace, block_map, tree_index_map)
    # Block-internal saved tensors only; the block-input residual lives in ``ckpt_chain_bytes``.
    saved_bytes_proxy_for_op_walk = _saved_tensor_bytes_per_block(trace)

    ckpt_bump_op: dict[int, int] = {}
    # OFFLOAD bump fires at the last forward op (closest to the block's backward window).
    offload_bump_op: dict[int, int] = {}
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)
        elif mode is BlockMode.OFFLOAD:
            offload_bump_op[op_idxs[-1]] = int(block_id)

    retained_none_bytes = 0
    # CKPT chain term factored to ``_compute_ckpt_chain_bytes`` so the wrapper-side
    # ``_reconstruct_f_bm`` shares the same chain-sum semantics.
    ckpt_chain_bytes = _compute_ckpt_chain_bytes(trace, block_map)
    for block_id_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(block_id_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            retained_none_bytes += act_sz
        # SWAP: live only during the block's forward compute; assumed
        #       to overlap free GPU memory (§3.3). The CKPT-chain term
        #       does NOT apply because SWAP evicts the block-output
        #       tensor to the pinned-CPU swap pool (see swap_pool.py).

    # --- Op walk -------------------------------------------------------
    raw_peak = 0
    # Track activations that are "live as of op i". We build this
    # incrementally so ops inside a NONE block see that block's
    # activation bytes accumulate progressively (safer upper bound even
    # though the end-of-fwd sum already accounts for all of it). The
    # simplest correct accounting is:
    #
    #   live_at_op = retained_none_bytes_accumulated_up_to_block(op)
    #              + ckpt_bump_if_this_op_triggers
    #
    # We pre-compute the cumulative "NONE activations active by this
    # point in forward" by walking blocks in order.

    # Map op index -> cumulative NONE-activation bytes active at or
    # before this op. Blocks without a position in forward_ops_by_block
    # contribute no ordering, so we sort blocks by their first forward
    # op index.
    block_first_op = {bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops}
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])

    cumulative_none: list[tuple[int, int]] = []  # (first_op_idx, cumulative_bytes)
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            # OFFLOAD retains forward activations on GPU (§3.3 lifecycle
            # table — "Forward activations: retained on GPU"). They join
            # the NONE running total so the live_none-at-op-i view sees
            # the same bytes as a NONE block would; the backward-window
            # chunk gather bump is a separate per-op bump landed via
            # ``offload_bump_op`` below.
            running += trace.activation_sizes.get(bid, 0)
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        """Cumulative NONE-block activation bytes at or before op_idx."""
        # Linear scan is fine; cumulative_none has at most N_block
        # entries (8-256 in realistic workloads).
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue

        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)

        # CKPT recompute bump = internal saved-tensor delta; block-input residual already in ``ckpt_chain_bytes``.
        ckpt_extra = 0
        if i in ckpt_bump_op:
            bid = BlockId(ckpt_bump_op[i])
            block_act = trace.activation_sizes.get(bid, 0)
            block_saved = int(saved_bytes_proxy_for_op_walk.get(bid, block_act))
            ckpt_extra = max(0, block_saved - block_act)

        # OFFLOAD bump = S_chunk gather only; activations already in live_none.
        offload_extra = 0
        if i in offload_bump_op:
            offload_extra = layout.S_chunk

        op_cross_attn = op_cross_attn_surcharge(op, cross_attn_bytes, tree_index_map)

        candidate = (
            model_state_present
            + live_none
            + ckpt_chain_bytes
            + ckpt_extra
            + offload_extra
            + op_cross_attn
            + intra
            + inter
        )
        if candidate > raw_peak:
            raw_peak = candidate

    # Degenerate trace (no forward ops): static estimate. ckpt_chain_bytes and retained_none_bytes are disjoint by construction so summing both does not double-count.
    if raw_peak == 0:
        raw_peak = model_state_present + retained_none_bytes + ckpt_chain_bytes

    # apply_hot_iter_cap bounds only the activation portion; preserves model-state through cap.
    measured_cap = hot_iter_peak_cap(trace, block_map, cfg, layout)
    raw_peak = apply_hot_iter_cap(raw_peak, model_state_present, measured_cap, layout)

    alpha = alpha_fragmentation_for_cfg(hw.dominant_param_bytes_per_element, cfg)
    scaled = int(alpha * raw_peak)
    LOG.debug(
        "estimate_peak: n_persist=%d n_buffer=%d n_swap=%d n_ckpt=%d n_offload=%d "
        "raw=%dB alpha=%.2f -> %dB",
        cfg.n_persist,
        cfg.n_buffer,
        cfg.n_swap,
        cfg.n_checkpoint,
        cfg.n_offload,
        raw_peak,
        alpha,
        scaled,
    )
    return scaled


__all__ = [
    "ALPHA_FRAGMENTATION",
    "ALPHA_FRAGMENTATION_4BIT",
    "ALPHA_FRAGMENTATION_4BIT_MODE_A",
    "ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT",
    "alpha_fragmentation_for_cfg",
    "alpha_fragmentation_for_dtype",
    "_compute_ckpt_chain_bytes",
    "_saved_tensor_bytes_per_block",
    "block_tree_index_map",
    "cross_attn_handoff_bytes",
    "cross_attn_persist_bytes",
    "estimate_cpu_footprint",
    "estimate_peak",
    "hot_iter_peak_cap",
    "model_state_present_bytes",
    "op_cross_attn_surcharge",
]
