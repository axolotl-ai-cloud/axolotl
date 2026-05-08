"""Peak-memory reconstruction for the ProTrain searcher (§3.3, App A.2).

Implements Eqs. 8-10 — an operator-by-operator walk of the forward pass
that tracks live tensors, adds the profiled intra- and inter-op deltas,
and accounts for the per-block activation strategy (NONE / CKPT / SWAP).
Applies Eq. 11 — the ``alpha`` fragmentation factor — as a final
multiplicative over-estimate so the searcher conservatively prunes.

Design contract (see DESIGN.md §Design Decisions):

- ``ALPHA_FRAGMENTATION = 1.10`` matches the paper's "up to 10%
  overestimate on best-selected configurations" claim.
- SWAP blocks do not contribute to the op-walk peak: the paper argues
  swap-in "only fires when memory is available", so activation swapping
  is assumed to trade runtime for zero steady-state peak.
- Gradient checkpointing bumps the peak at the *first* op of each CKPT
  block — this is when recomputation materializes the block's
  activations before the backward pass consumes them.
- ZeRO-3 sharding (``HardwareProfile.zero3_shard=True``) does NOT
  reduce the GPU peak: each rank's gather issues
  ``all_gather_into_tensor`` to reconstruct the full chunk on GPU
  before forward/backward compute, so the buffer-pool residency term
  is identical to the replicated path. Sharding only changes the
  per-rank pinned CPU footprint — see :func:`estimate_cpu_footprint`.
- The persistent-chunk and buffer-slot multipliers in
  :func:`estimate_peak` are derived from
  ``trace.model_state_bytes / (N_chunk * S_chunk)`` rather than the
  raw fp16 param size: ``S_chunk`` itself is computed from fp16 PARAM
  bytes only (see :func:`axolotl.integrations.protrain.chunk.layout._param_bytes`),
  so a persistent chunk's true GPU residency under full fp16 + Adam
  is ~8x ``S_chunk`` (params + grads + fp32 master + 2x momenta), not
  1x. See the inline derivation at the top of :func:`estimate_peak`.
"""

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


#: Module-level latch: ``model_state_present_bytes`` is called once per
#: candidate inside the search loop, so emitting the stale-trace warning
#: directly would fire it once per candidate (search spaces are O(10²)+).
#: Gate the warning on this flag so it fires exactly once per process.
_STALE_TRACE_WARNING_EMITTED = False


def _saved_tensor_bytes_per_block(trace: ProfilerTrace) -> dict[BlockId, int]:
    """Per-block saved-for-backward bytes proxy from steady-state forward peaks.

    ``trace.activation_sizes[bid]`` records ONLY the block's OUTPUT tensor
    bytes (~2 MB on 7B Llama). The actual saved-tensor footprint a block
    leaves resident across forward (Q/K/V/output projections + MLP
    intermediate states + attention scores) is ~30x larger (~60 MB on
    7B Llama). When the cap subtracts ``activation_sizes`` for CKPT/SWAP
    blocks it under-credits the savings by exactly that factor, leaving
    the predicted peak insensitive to ``n_checkpoint`` (the headline 7B
    test sees a 13% over-prediction that's ~30x explained by this gap).

    Reconstruct a more faithful per-block proxy from
    ``steady_fwd_block_peak_bytes`` — the block-level
    ``max_memory_allocated`` snapshots taken during the hook-less steady
    forward (peak counter reset between blocks). Within each block's
    window, ``peak[bid] = allocated_at_block_start + intra_op_peak``, and
    ``allocated_at_block_start`` rises monotonically with cumulative
    saved-tensor bytes of preceding blocks. So ``peak[bid] - peak[bid-1]``
    ≈ block ``bid-1``'s saved-tensor bytes (intra peaks cancel under a
    uniform-block assumption). We attribute that forward difference to
    block ``bid-1`` and use the median forward difference as a fallback
    for block 0 (no predecessor to diff against).

    Returns a dict mapping ``BlockId -> bytes``. Falls back to
    ``trace.activation_sizes[bid]`` for any block where the per-block
    peak data is missing or yields a non-positive delta. Empty when
    neither source is populated.
    """
    deltas: dict[BlockId, int] = {}
    persisted_per_block_peak = getattr(trace, "steady_fwd_block_peak_bytes", None) or {}
    # Normalize keys + values back through ``BlockId(int(...))`` /
    # ``int(...)`` here, BEFORE the sorted/lookup paths below — so a
    # cached trace whose keys round-tripped through JSON / pickle as
    # strings still hits ``per_block_peak.get(prev_bid)`` (which uses
    # the coerced ``BlockId`` type). Without this normalization the
    # ``.get`` calls miss on every lookup and the helper silently
    # falls back to ``activation_sizes``, disabling the saved-tensor
    # proxy on reloaded traces.
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

    # Fill any remaining gaps from ``activation_sizes`` (e.g. blocks that
    # appear in ``activation_sizes`` but not in ``steady_fwd_block_peak_bytes``,
    # or blocks where the forward diff was zero / negative).
    for bid_raw, act_sz in activation_sizes.items():
        bid = BlockId(int(bid_raw))
        if bid not in deltas:
            deltas[bid] = int(act_sz)

    return deltas


#: Eq. 11 fragmentation factor — applied as a final multiplier on the
#: raw op-walk peak. Treated as a module-level constant so tests can
#: import it explicitly for sanity checks.
#: Matches the paper's "up to 10% overestimate on best-selected
#: configurations" claim. Previously bumped to 1.20 as an empirical
#: band-aid for backward-peak underprediction; with the M4.5 runtime
#: gaps now closed (per-param grad offload, init-time chunk offload,
#: the BUG-1-4 fixes in ``chunk/manager.py``) the op-walk matches
#: measured peaks tightly enough to restore the paper value — see
#: DESIGN.md §Design Decisions point 1.
ALPHA_FRAGMENTATION: float = 1.10


def _group_ops_by_block(trace: ProfilerTrace) -> dict[BlockId, list[int]]:
    """Return ``{block_id -> [op_positions]}`` for forward ops only.

    ``op_positions`` are indices into ``trace.op_order``; ops that do
    not belong to any block (e.g. embedding, final LM head) are skipped.
    """
    grouped: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        if op.block_id is None:
            continue
        grouped[op.block_id].append(i)
    return grouped


def _tree_index_for_path(module_path: str) -> int:
    """Best-effort tree-index inference from a module path.

    Tree boundaries are not stored in ``ProfilerTrace`` directly, so we
    parse the dotted path's first segment:

    - ``encoder...`` -> tree 0
    - ``decoder...`` -> tree 1
    - anything else  -> tree 0 (single-tree default)

    This mirrors the convention used by
    :func:`axolotl.integrations.protrain.block.layout_rules.flatten_block_trees`,
    which gives the encoder ``forward_order=0`` and the decoder
    ``forward_order=1``. Single-tree causal-LM models have all paths
    fall through to tree 0, preserving legacy behaviour exactly.

    The two-tree case targets T5 / FLAN-T5 (Item 9). BART would also
    classify correctly here — its block paths are ``encoder.layers``
    / ``decoder.layers``. Future enc-dec families with non-``encoder``/
    ``decoder`` naming would need explicit handling.
    """
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
    """Map each ``BlockId`` to its forward-order tree index.

    Reads ``trace.block_tree_index`` when populated (TRACE_VERSION ≥ 12,
    where the trace constructor walks ``discover_blocks(model)`` and
    records ``block_id -> forward_order`` directly). Falls back to
    parsing the first forward op's ``module_path`` prefix (``encoder.``
    -> 0, ``decoder.`` -> 1, else 0) for degenerate test inputs that
    don't carry the field. Returns ``{}`` if no forward ops carry
    block_ids and the persisted map is empty.
    """
    persisted = getattr(trace, "block_tree_index", None)
    if persisted:
        # Defensive normalisation: the cached map may round-trip
        # through JSON / pickle that stringifies dict keys, while the
        # rest of this module looks them up via ``BlockId(int(...))``.
        # Mirror the coercion the per-block loop in ``estimate_peak``
        # already applies so the encoder/decoder surcharge path
        # doesn't silently disable on a re-loaded trace whose keys
        # are strings.
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
    """Estimate cross-attention saved-state bytes that span trees.

    Encoder-decoder models (T5, FLAN-T5) save the encoder's last-layer
    hidden state for cross-attention in the decoder. That tensor is
    produced during encoder forward, consumed during decoder forward
    (every cross-attention layer reads it), and released only after
    decoder backward finishes — so it spans the entire decoder
    forward + decoder backward window.

    Sizing — interpretation of T5's saved-state, NOT covered by the
    paper (paper is causal-LM only):

    - Use ``activation_sizes[last_enc_bid]`` as a CONSERVATIVE upper
      bound. The retained-activation-bytes value for the encoder's
      final block already includes the hidden-state output that gets
      passed to the decoder; it's strictly larger than the
      cross-attn-only saved-state.
    - When that block is in NONE or OFFLOAD mode the bytes are already
      counted in :func:`estimate_peak`'s ``live_none`` accumulator
      (OFFLOAD retains forward activations on GPU symmetrically to
      NONE — see the ``retained_none_bytes`` / ``cumulative_none``
      construction below), so we return ``0`` to avoid double-counting.
    - When that block is in CKPT or SWAP mode its activations are not
      in ``live_none``; CKPT discards the BLOCK INTERNALS but the
      OUTPUT hidden tensor passed to the decoder cannot be discarded
      (the cross-attention layers reference it). Same for SWAP — the
      saved-state output isn't part of the swap-band's offload set.
      We therefore return the full ``activation_sizes`` upper bound.

    Returns 0 when the trace looks single-tree (no decoder ops), when
    no encoder block_ids resolve, or when we lack activation bytes for
    the last encoder block.
    """
    if not _has_multiple_trees(tree_index_map):
        return 0
    encoder_bids = sorted(bid for bid, idx in tree_index_map.items() if idx == 0)
    if not encoder_bids:
        return 0
    last_enc_bid = encoder_bids[-1]
    last_enc_mode = block_map.get(last_enc_bid, BlockMode.NONE)
    if last_enc_mode is BlockMode.NONE or last_enc_mode is BlockMode.OFFLOAD:
        # Already counted in retained_none_bytes; avoid double-counting.
        # OFFLOAD retains forward activations on GPU like NONE (the
        # OFFLOAD-only bump is the per-block backward chunk gather,
        # tracked separately via ``offload_bump_op`` in estimate_peak).
        return 0
    return int(trace.activation_sizes.get(last_enc_bid, 0))


def op_cross_attn_surcharge(
    op: OpRecord,
    cross_attn_bytes: int,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Per-op cross-attention surcharge during decoder forward.

    Returns ``cross_attn_bytes`` if this op belongs to a non-encoder
    tree (decoder forward); ``0`` otherwise. Shared by
    :func:`estimate_peak` and the searcher fast-path
    :func:`axolotl.integrations.protrain.search.exhaustive._block_map_peak_contribution`
    so both walks gate identically on the tree index.
    """
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
    """Measured ground-truth upper bound on the raw op-walk peak, or None.

    Prefers per-block data from TRACE_VERSION ≥ 6:
    ``max(steady_fwd_block_peak_bytes) + max_ckpt_activation
    + offload_bump`` under the given ``block_map``. Falls back to the
    aggregate ``steady_fwd_peak_bytes`` (v5) but only when ``cfg`` is
    provided AND the config is fully-NONE (the aggregate makes no
    provision for CKPT recomp / OFFLOAD gather bumps). Returns ``None``
    when no hot-iter data is available — callers then leave the op-walk
    raw peak untouched.

    OFFLOAD bump (R5-A): :func:`estimate_peak` adds an ``S_chunk``
    surcharge at the LAST forward op of each OFFLOAD block (the
    backward-window chunk-gather; see Option B §4.1). Because OFFLOAD
    bumps fire one-at-a-time across the op-walk (each at a different op
    index) and the searcher's peak takes the per-op maximum, only ONE
    such ``S_chunk`` bump contributes to the modeled peak — analogous
    to how the CKPT bump adds ``max_ckpt_activation`` once. The
    steady-forward profiling pass that produces
    ``steady_fwd_block_peak_bytes`` runs under the all-NONE policy and
    therefore captures none of the OFFLOAD chunk-gather residency. We
    must add it back here, otherwise the cap clamps OFFLOAD configs
    below their own modeled peak (the searcher would then over-prefer
    OFFLOAD configs that don't actually fit). Requires ``layout`` to be
    threaded through; when ``layout`` is ``None`` (legacy callers) the
    OFFLOAD bump degrades to ``0`` and the cap behaviour matches the
    pre-R5-A implementation — the legacy fallback never activates from
    in-tree call sites, which all pass ``layout``.

    Used by BOTH :func:`estimate_peak` (full op-walk path) and
    :func:`axolotl.integrations.protrain.search.exhaustive.search`
    (inline F_bm fast path) so the cap propagates to the searcher's
    picked config, not just to ``estimate_peak`` callers.
    """
    if trace.steady_fwd_block_peak_bytes:
        forward_max_block_peak = max(trace.steady_fwd_block_peak_bytes.values())
        # Backward-aware cap (TRACE_VERSION ≥ 19): when the steady
        # backward measurement is available, use the larger of the
        # per-block forward and per-block backward peaks. The bootstrap
        # backward holds gradient buffers + saved-for-backward
        # activations live simultaneously across each block's bwd
        # window, which can exceed that block's forward peak; folding
        # the bwd-side max into the cap keeps the cap an honest upper
        # bound for any candidate (NONE/CKPT/SWAP) block_map. When the
        # bwd field is empty (older trace, on-demand engaged, bwd iter
        # raised) this degrades to forward-only — preserves v18
        # behaviour.
        bwd_per_block = getattr(trace, "steady_bwd_block_peak_bytes", None) or {}
        if bwd_per_block:
            backward_max_block_peak = max(bwd_per_block.values())
            forward_max_block_peak = max(
                forward_max_block_peak, backward_max_block_peak
            )
        ckpt_recomp_bump = 0
        has_offload = False
        # n_checkpoint / n_swap savings (Fix 1).
        #
        # The per-block peaks in ``steady_fwd_block_peak_bytes`` /
        # ``steady_bwd_block_peak_bytes`` were captured under the
        # profiler's hook-less steady run, which does NOT wrap blocks
        # for activation checkpointing — every block's forward retains
        # its full saved-tensor set as if mode were NONE. The block
        # post-hook fires AFTER that block's forward, so the LAST
        # block's per-block peak ≈ profile-time model state +
        # cumulative_NONE(all_blocks) + intra_last. ``forward_max_block_peak``
        # is therefore the all-NONE peak ceiling.
        #
        # When the production block_map has CKPT or SWAP blocks, those
        # blocks' activation bytes are NOT retained at runtime peak:
        #   * CKPT block discards its forward saved tensors and
        #     rematerializes them at backward (one block at a time);
        #     the recompute window contributes ``ckpt_recomp_bump`` which
        #     we add back below.
        #   * SWAP block evicts its saved tensors to a pinned-CPU pool
        #     and the pool slot is decoupled from GPU residency; the
        #     paper §3.3 treats SWAP as zero steady-state GPU peak.
        #
        # Subtract their ``activation_sizes`` aggregate from the all-NONE
        # ceiling so the cap shrinks linearly with n_checkpoint and n_swap,
        # tracking the actual GPU residency the production runtime delivers.
        # Without this subtraction the cap is a flat all-NONE ceiling that
        # over-predicts equally for every n_checkpoint > 0 (the bug Fix 1
        # addresses: predicted peak identical for n_checkpoint=1 and
        # n_checkpoint=9 even though actual peak drops with k).
        #
        # Magnitude proxy (Fix 2). ``activation_sizes[bid]`` is the
        # block's OUTPUT-bytes proxy (residual stream + logits, ~2 MB
        # on 7B Llama) — too small by ~30x for the savings calculation:
        # the actual saved-tensor footprint a NONE block leaves resident
        # (Q/K/V/output projections + MLP intermediate states + attention
        # scores) is ~60 MB on 7B Llama. Subtracting the output-only
        # proxy makes the cap shrink by ~2 MB per CKPT block when the
        # actual production peak shrinks by ~60 MB per CKPT block — a
        # 30x miscalibration that surfaces as a 13% over-prediction on
        # the headline 7B test (9 blocks checkpointed: predicted drops
        # 18 MB, actual drops ~540 MB).
        #
        # Use the per-block forward-peak deltas via
        # :func:`_saved_tensor_bytes_per_block` as the savings proxy:
        # those deltas measure the cumulative-allocated rise between
        # adjacent blocks during the hook-less steady forward, which IS
        # the saved-for-backward residency the production runtime
        # actually frees when the block is wrapped for CKPT/SWAP. Falls
        # back to ``activation_sizes`` per-block when per-block peak
        # data is missing (older traces; gaps in capture) so this fix
        # is a strict upgrade over the previous behaviour — the cap
        # tightens for any trace carrying ``steady_fwd_block_peak_bytes``
        # without breaking callers that don't.
        #
        # ``ckpt_recomp_bump`` continues to use ``activation_sizes`` —
        # that term models the per-block recomputation peak during
        # backward (paper §3.3: "one block at a time, serially"), where
        # the dominant cost is the block's output materialization in
        # the recomp window. Saved-for-backward bytes are released
        # before the recomp window opens, so the larger
        # saved-tensor-bytes proxy doesn't apply here. Only the
        # savings (what NONE retains across the whole forward but CKPT
        # / SWAP do not) need the larger proxy.
        saved_bytes_proxy = _saved_tensor_bytes_per_block(trace)
        # Encoder→decoder handoff fence (encoder-decoder traces only).
        #
        # ``estimate_peak`` re-adds ``cross_attn_persist_bytes(...)`` as
        # a per-decoder-op surcharge when the encoder's last block is
        # CKPT or SWAP, because the cross-attention output tensor must
        # remain GPU-resident across the whole decoder fwd+bwd window
        # even though the rest of the encoder block's saved tensors
        # have been discarded / swapped (see ``cross_attn_persist_bytes``
        # docstring at line 232 and the ``op_cross_attn_surcharge`` op-walk
        # gating at lines 286-303). Subtracting that block's full
        # ``block_saved`` here would double-discount: the cap shrinks by
        # the full saved-tensor bytes while the runtime peak only
        # actually drops by ``block_saved - cross_attn_persist_bytes``
        # (the encoder→decoder hidden tensor stays live). Cap the
        # encoder-last block's contribution to ``max(0, block_saved -
        # cross_attn_persist_bytes)`` for CKPT/SWAP modes; non-encdec
        # traces are unaffected because ``cross_attn_persist_bytes``
        # returns 0 outside the multi-tree path.
        tree_index_map = block_tree_index_map(trace)
        cross_attn_bytes_for_cap = cross_attn_persist_bytes(
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
                # Cap the encoder-last savings by the persisted handoff
                # tensor — the cross-attention output cannot be
                # reclaimed on this block under CKPT/SWAP.
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
        # Floor the activation portion at zero (subtract-back guards): if
        # the cumulative ``ckpt_swap_savings`` exceed the gap between the
        # all-NONE ceiling and the profile-time model state, treat the
        # activation portion as zero. ``apply_hot_iter_cap`` re-attaches
        # ``model_state_present`` on top — so the final cap collapses to
        # the model-state baseline + the ckpt_recomp bump in the limit
        # where every block is CKPT/SWAP.
        profile_time_model_state = (
            layout.N_chunk * layout.S_chunk if layout is not None else 0
        )
        # The all-NONE ceiling decomposes into model-state + activation
        # portion; only the activation portion shrinks with CKPT/SWAP.
        # Clamp the savings so they cannot drive the cap below the
        # model-state floor. Then add back the recomp / gather bumps.
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
    # Aggregate fallback path. ``steady_bwd_peak_bytes`` (TRACE_VERSION
    # ≥ 19) typically exceeds ``steady_fwd_peak_bytes`` because backward
    # holds grads + saved activations simultaneously. Take the max so
    # all-NONE configs see the tighter measured ceiling.
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
    """Apply :func:`hot_iter_peak_cap` to ``raw_peak`` using layered decomposition.

    ``measured_cap`` is the profiler's hook-less steady FORWARD peak
    (``trace.steady_fwd_peak_bytes`` / max ``steady_fwd_block_peak_bytes``).
    That capture happens BEFORE the optimizer is constructed — only fp16
    params + the forward's max activation are resident at that moment.
    Therefore the cap decomposes as
    ``measured_cap = profile_time_model_state + measured_activation_cap``
    where ``profile_time_model_state == layout.N_chunk * layout.S_chunk``
    (full fp16 param set; on-demand mode skips this capture entirely so
    the cap is ``None`` there and this helper short-circuits).

    Layered cap (post-d908bf28 fix): cap ONLY the activation portion of
    ``raw_peak``, leaving ``model_state_present`` (which can scale up to
    ~8x for full FT through Adam state) intact through the cap. Applying
    the cap to the full ``raw_peak`` (the pre-fix shape) silently erased
    the per-chunk Adam-state contribution and produced ~90x under-
    predictions on full-FT shapes (Codex confirmed: estimate_peak ~7 GB
    vs. raw clamp ~78 MB on the same config).

    Returns the post-cap raw_peak. When ``measured_cap is None`` returns
    ``raw_peak`` unchanged so callers can use this helper unconditionally.

    Used by both :func:`estimate_peak` (full op-walk path) and
    :func:`axolotl.integrations.protrain.search.exhaustive.search`'s
    inline F_bm fast path so the two sites cannot drift again.
    """
    if measured_cap is None:
        return raw_peak
    # Decompose raw_peak. Defensive max() guards degenerate traces where
    # retained_none_bytes < model_state_present (op-walk skipped because
    # no forward ops); the static fallback in estimate_peak already set
    # raw_peak = model_state_present + retained_none_bytes, so the
    # subtraction is non-negative in practice. The searcher's inline path
    # constructs raw_peak as ``model_state_present + f_bm`` and f_bm is
    # always non-negative, so this is also safe there.
    op_walk_portion = max(0, raw_peak - model_state_present)
    # Decompose measured_cap. The profiler ran with all params resident
    # on GPU whenever this cap is populated — on-demand mode skips the
    # entire steady-fwd capture (``profiler/trace.py::run_trace`` gates
    # the per-block + aggregate measurement on ``not engage_on_demand``),
    # so ``hot_iter_peak_cap`` returns ``None`` and the early-return
    # above fires. Therefore the captured peak always includes
    # ``N_chunk * S_chunk`` of resident fp16 params; subtract to recover
    # the activation ceiling. Clamp at 0 so synthetic test traces
    # (e.g. ``test_estimate_peak_uses_per_block_caps`` sets a per-block
    # peak smaller than the layout's fp16 total) don't yield a negative
    # cap — in that degenerate case the activation portion is pinned to
    # 0 and raw_peak collapses to the model-state floor.
    profile_time_model_state = layout.N_chunk * layout.S_chunk
    measured_activation_cap = max(0, measured_cap - profile_time_model_state)
    if op_walk_portion > measured_activation_cap:
        op_walk_portion = measured_activation_cap
    # Reassemble: model_state_present is preserved through the cap.
    return model_state_present + op_walk_portion


#: Pool sizing knobs mirrored from ``block.swap_pool.ActivationSwapPool``.
#: The pool holds ``n_swap * SWAP_SLOTS_PER_BLOCK * SWAP_PREFETCH_DEPTH``
#: activation slots, each sized to the worst-case single-saved-tensor
#: bytes across the swap-band. Kept in sync with the wrapper's defaults
#: (single-block lookahead = 2; K=8 saved tensors per block forward).
#: When tuning these, update both these constants AND the
#: model_wrapper's ``ActivationSwapPool(prefetch_depth=..., slots_per_block=...)``
#: arguments so the cost model reflects the runtime pool sizing.
SWAP_PREFETCH_DEPTH: int = 2
SWAP_SLOTS_PER_BLOCK: int = 8


def estimate_cpu_footprint(
    cfg: CostConfig,
    layout: ChunkLayout,
    hw: HardwareProfile,
    trace: ProfilerTrace | None = None,
) -> int:
    """Per-rank pinned CPU bytes held by non-persistent chunks + SWAP slots.

    The non-persistent chunks live on CPU in pinned memory. Under the
    replicated (pre-M7) path every rank holds a FULL copy of each
    non-persistent chunk, so the per-rank footprint is
    ``(N_chunk - n_persist) * S_chunk``. Under the M7 ZeRO-3 sharded
    path each rank holds only ``ceil(chunk_bytes / world_size)`` per
    chunk, so the per-rank footprint divides by ``gpu_count``.

    The activation-swap pool, when ``n_swap > 0`` and a trace is
    provided, contributes an additional
    ``n_swap * SWAP_SLOTS_PER_BLOCK * SWAP_PREFETCH_DEPTH * slot_bytes``
    of pinned CPU, where ``slot_bytes`` is the per-block AGGREGATE
    activation bytes (NOT divided by ``SWAP_SLOTS_PER_BLOCK``). The
    trace records only the per-block aggregate — there is no per-saved-
    tensor breakdown — and real transformer blocks have skewed tensor
    size distributions where the residual stream alone can dominate
    ~1/3-1/2 of the aggregate. Sizing slots to the average would let
    the runtime ``ActivationSwapPool`` raise ``RuntimeError`` whenever
    SWAP encountered a single saved tensor larger than the average.
    Sizing every slot to the full aggregate over-provisions the pool
    by up to Kx but guarantees any saved tensor fits any slot — see
    the matching slot-sizing comment in
    ``api/model_wrapper.py::_construct_runtime`` for the runtime side.
    The term is **per-rank** and **NOT divided by gpu_count** — the
    swap pool is a rank-local allocation; sharding does not split
    activations across ranks. The conservative-upper-bound contract
    the searcher gate expects is preserved (this term is now strictly
    larger than the previous average-derived estimate). When ``trace``
    is None we omit the swap term — used by
    callers that want a pre-search ballpark; the searcher itself
    always passes ``trace`` so the gate matches the real wrap-time
    pool size.

    This accounting is **orthogonal to** :func:`estimate_peak`, which
    models GPU memory: the gather materializes the full chunk on GPU
    via ``all_gather_into_tensor`` regardless of sharding, so GPU peak
    is unchanged by ``zero3_shard``. The real savings from sharding
    appear here (CPU bytes/rank) and in the reduce bandwidth
    (reduce_scatter vs. per-param all_reduce).

    Parameters
    ----------
    cfg:
        Candidate knob configuration. ``n_persist`` controls the chunk
        contribution; ``n_swap`` controls the activation-swap term.
        ``n_buffer``/``n_checkpoint`` never change pinned CPU footprint.
    layout:
        Chunk layout. ``S_chunk`` and ``N_chunk`` are read directly.
    hw:
        Hardware profile. Reads ``gpu_count`` and ``zero3_shard``.
    trace:
        Optional profiler trace. When provided, the activation-swap
        term uses the actual swap-band's max activation bytes
        (``max(activation_sizes[bid])`` over the first ``n_swap``
        blocks under the swap-early rule from ``assign_modes``). When
        absent and ``n_swap > 0``, returns the chunk term only — used
        by older callers that don't have a trace handle. The searcher
        always passes the trace so its feasibility gate is precise.

    Returns
    -------
    int
        Per-rank pinned CPU bytes. Rounded up via ceiling division so
        the returned value is a conservative upper bound on actual
        shard allocations (shard sizes themselves are rounded up to a
        dtype-aligned boundary by ``ChunkManager.materialize_offload``;
        the arithmetic here tracks the same ceiling).
    """
    # Non-persistent chunks are everything outside the augmented
    # persistent set (prefix union mandatory_persistent). Earlier this used
    # ``N_chunk - cfg.n_persist``, which double-counted any
    # ``layout.mandatory_persistent`` chunk as a CPU shard even though
    # the runtime never offloads it. With non-empty ``mandatory_persistent``
    # this would over-charge the per-rank pinned-CPU budget by up to
    # ``len(mandatory_persistent) * S_chunk``.
    n_persist_eff = len(layout.effective_persistent_ids(cfg.n_persist))
    non_persist = max(0, layout.N_chunk - n_persist_eff)
    # Under sharding each rank holds 1/world of each chunk — ZeRO-3
    # partitions across the FULL distributed world, not just the GPUs
    # on the local node. ``hw.gpu_count`` is the LOCAL device count
    # (typically the GPUs on this node) and would under-divide the
    # per-rank shard in any multi-node config, materially overstating
    # the per-rank pinned CPU footprint and pessimizing the searcher's
    # feasibility gate. ``trace.world`` is the world size recorded at
    # profile time (resolved from ``torch.distributed`` if initialized,
    # else 1), which is the correct partitioning denominator. Fall
    # back to ``hw.gpu_count`` when no trace is supplied — that path
    # is single-node by construction (the trace-less callers are pre-
    # search ballparks). Ceiling division is applied PER CHUNK so
    # small chunks don't underreport when ``S_chunk`` isn't divisible
    # by ``per_rank_divisor`` — summing ``total_bytes`` first and
    # dividing once would round only at the aggregate, undercounting
    # the trailing-rank padding by up to ``non_persist - 1`` bytes.
    if hw.zero3_shard:
        per_rank_divisor = trace.world if trace is not None else hw.gpu_count
    else:
        per_rank_divisor = 1
    per_rank_divisor = max(1, per_rank_divisor)
    per_chunk_sharded = (layout.S_chunk + per_rank_divisor - 1) // per_rank_divisor
    chunk_term = non_persist * per_chunk_sharded

    # Activation-swap pool term — rank-local; not sharded.
    #
    # The runtime pool (``block.swap_pool.ActivationSwapPool``) reserves
    # ``n_swap * SWAP_SLOTS_PER_BLOCK * SWAP_PREFETCH_DEPTH`` pinned CPU
    # slots, each sized to the worst-case single-saved-tensor bytes.
    # The trace exposes only the per-block AGGREGATE
    # (``activation_sizes[bid]``); a single saved tensor inside that
    # block can be a large fraction of the aggregate (residual stream)
    # so dividing by ``SWAP_SLOTS_PER_BLOCK`` would underestimate the
    # required slot width and let the runtime ``slot_view.copy_(tensor)``
    # raise. Until per-saved-tensor profiling lands, size each slot to
    # the full per-block aggregate — a strict upper bound that matches
    # the matching slot-sizing branch in
    # ``api/model_wrapper.py::_construct_runtime``.
    swap_term = 0
    if cfg.n_swap > 0 and trace is not None and trace.activation_sizes:
        # Swap-early rule: the first ``n_swap`` blocks (in BlockId order)
        # use SWAP.
        sorted_bids = sorted(trace.activation_sizes.keys())
        swap_band = sorted_bids[: cfg.n_swap]
        if swap_band:
            per_block_activation_bytes = max(
                int(trace.activation_sizes.get(bid, 0)) for bid in swap_band
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
    """Resident model-state bytes for the runtime's persistent + buffer set.

    The runtime persistent set is
    ``layout.effective_persistent_ids(cfg.n_persist)`` —
    the user-chosen prefix ``[0, n_persist)`` UNIONED with
    ``layout.mandatory_persistent`` (chunks the block-granularity
    scheduler cannot gather on its own; pinned for runtime correctness,
    NOT chosen by the searcher). The cost charged here is therefore
    ``len(prefix union mandatory) * S_chunk * persistent_factor``, NOT
    ``cfg.n_persist * S_chunk * persistent_factor``.

    Sums the per-chunk persistent contribution (full fp16 + grads + fp32
    master + Adam moments under full FT, just fp16 params under
    LoRA-with-frozen-base) plus the per-chunk transient buffer-pool
    contribution (fp16 params gathered + fp16 grads accumulated during
    backward). Centralized here so the searcher's inline pruning
    (``search/exhaustive.py``) and the post-search validator
    (:func:`estimate_peak`) cannot drift in their model-state accounting
    — see paper Eq. 11 derivation in :func:`estimate_peak`'s body for
    the per-factor justification.

    When ``trace.model_state_bytes`` is missing or zero, falls back to
    the legacy ``persistent_factor = 1.0`` (paper's implicit
    "params-only on GPU" assumption — strictly an UNDER-estimate for
    full FT) and emits a one-shot warning so the regression is visible.
    """
    persistent_ids = layout.effective_persistent_ids(cfg.n_persist)
    n_persist_eff = len(persistent_ids)
    # n_buffer is bounded by the non-persistent chunk count (everything
    # outside the augmented persistent set is eligible for buffer-pool
    # caching). Earlier this was ``N_chunk - cfg.n_persist`` which
    # over-counted available slots when ``mandatory_persistent`` was
    # non-empty — a buffer slot for a mandatory chunk is dead weight.
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist_eff))

    fp16_total_bytes = layout.N_chunk * layout.S_chunk
    model_state_total = int(getattr(trace, "model_state_bytes", 0) or 0)
    if fp16_total_bytes > 0 and model_state_total > 0:
        # Per-chunk multiplier: aggregate state bytes / fp16-params total.
        # Clamp >= 1.0 because the aggregate by construction includes
        # the fp16 params themselves (any value < 1.0 would indicate a
        # trace bug; we still respect it but never go below the legacy
        # behaviour).
        persistent_factor = max(1.0, model_state_total / fp16_total_bytes)
    else:
        # Warn-once gate: this function runs inside the search loop, so a
        # bare ``LOG.warning`` would fire once per candidate. The latch is
        # process-scoped — fine for a single search invocation; tests
        # that need to re-trigger the warning can reset the flag explicitly.
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
    # Buffer slot during backward = fp16 params (gathered) + fp16 grads
    # (accumulated). 2.0 is a strict upper bound on the buffer pool's
    # transient peak; the optimizer state never lives in the buffer
    # pool itself (non-persistent chunks have their fp32 master + m +
    # v on CPU and only stream onto GPU through the optimizer's own
    # transient buffers, accounted for separately by the runtime model).
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
    hw: HardwareProfile,  # noqa: ARG001 - accepted for API symmetry with runtime
) -> int:
    """Estimate steady-state peak GPU memory in bytes.

    Walks ``trace.op_order`` in forward order. At each op the candidate
    peak is:

        model_state_present
        + activations_live_at_op
        + intra_op_delta[op]
        + inter_op_delta[op_prev -> op]

    Then scaled by ``ALPHA_FRAGMENTATION``. See module docstring for the
    SWAP / CKPT accounting rules.

    Parameters
    ----------
    cfg:
        Candidate knob configuration. Only ``n_persist`` and
        ``n_buffer`` are consumed directly here; ``n_swap`` and
        ``n_checkpoint`` show up via ``block_map``.
    trace:
        Output of the M1 profiler. Provides op order, intra/inter deltas,
        per-block activation sizes.
    layout:
        Chunk layout (``S_chunk``, ``N_chunk``).
    block_map:
        Per-block mode assignment (output of ``assign_modes``).
    hw:
        Hardware profile — currently unused, accepted for API symmetry
        with ``estimate_runtime`` so the searcher can call both with the
        same argument pack.

    Returns
    -------
    int
        Peak bytes, rounded via ``int(alpha * raw_peak)``.

    Notes — encoder-decoder peak accounting (Fix 3, post-Item 9)
    ------------------------------------------------------------
    The paper's §3.3 op-walk derivation assumes a single transformer
    tree (causal-LM); it does not cover encoder-decoder models. Our
    interpretation, applied transparently when the trace has both
    ``encoder.*`` and ``decoder.*`` ops:

    1. **Per-tree forward order:** the trace's ``op_order`` already
       interleaves the trees in their forward execution sequence
       (encoder first, then decoder), because
       ``flatten_block_trees`` numbers encoder block_ids before decoder
       ones, and the profiler trace tags ops with these global ids.
       The single op-walk below therefore traverses the trees in the
       correct order without further restructuring.
    2. **Cross-attention saved-state term:** the encoder's final hidden
       state lives across the entire decoder forward + decoder backward
       window. When the encoder's last block is in CKPT/SWAP mode its
       full activation bytes are not in ``live_none``, but the output
       hidden tensor still IS retained for cross-attn — so we add
       ``cross_attn_persist_bytes`` as a per-decoder-op surcharge.
       When the encoder's last block is NONE or OFFLOAD the bytes are
       already in ``live_none`` (OFFLOAD retains forward activations
       on GPU like NONE); the helper returns 0 to avoid double-counting.
    3. **Backward sequencing:** decoder backward runs to completion
       before encoder backward starts. The forward-driven peak we
       compute here is naturally an upper bound on the backward peak
       in this regime — at the last forward op every NONE activation
       across both trees plus the cross-attn saved state is live, and
       backward only frees them. The CKPT recomputation bump remains
       a forward-op surcharge as before, modeling the worst single
       block's recompute window.

    For single-tree causal-LM traces ``_has_multiple_trees`` is False,
    the cross-attn term is 0, and the op-walk is bit-identical to the
    pre-Fix-3 implementation. This is asserted by the cost-model unit
    tests in ``test_cost_search.py``.
    """
    # --- Static model-state footprint ----------------------------------
    # Persistent chunks are always on GPU. Non-persistent chunks only
    # occupy GPU memory through the buffer pool, so their GPU residency
    # is ``n_buffer * S_chunk`` not ``(N_chunk - n_persist) * S_chunk``.
    # Clamp n_persist/n_buffer into [0, N_chunk] defensively — the
    # searcher should never violate these, but other callers may.
    #
    # FULL-STATE ACCOUNTING (paper Eq. 11). The raw ``S_chunk`` is
    # derived in ``chunk/layout.py`` from fp16 PARAM bytes only — it
    # carries no grads, no fp32 master, and no Adam moments. A
    # persistent chunk under full fp16 + Adam fine-tune actually pins
    # the full per-param state on GPU:
    #
    #     fp16 params (1xS) + fp16 grads (1xS) + fp32 master (2xS)
    #         + fp32 exp_avg (2xS) + fp32 exp_avg_sq (2xS) ~= 8xS
    #
    # ``trace.model_state_bytes`` (set by
    # ``profiler/trace.py::_count_model_state_bytes``) carries this
    # aggregate (frozen-param resident bytes + per-trainable-param
    # 4-byte param+grad + 12-byte fp32-master+m+v). Dividing by the
    # fp16 chunk total ``N_chunk * S_chunk`` recovers the per-chunk
    # multiplier:
    #
    # * Full FT (every param trainable, fp16 + Adam): factor ~= 8.0
    # * LoRA with frozen base (Adam state only on the tiny adapter
    #   set): factor ~= 1.0 (frozen params dominate the aggregate so
    #   model_state_bytes ~= fp16 param bytes).
    #
    # A buffer slot only holds the transient gather + grad accumulation
    # during the backward window, NOT the optimizer state (which lives
    # only on the chunks the optimizer is currently stepping — handled
    # by the runtime cost model, not the peak model). So the buffer
    # coefficient is fixed at 2.0 (fp16 params + fp16 grads). This is
    # the same conservative coefficient the runtime materializes during
    # backward — see ``chunk/manager.py::gather`` + the per-param grad
    # offload path in ``api/model_wrapper.py``.
    #
    # When ``trace.model_state_bytes`` is unset/zero (older traces
    # predating this field's population), fall back to the legacy
    # 1xS_chunk multiplier and log a warning so the searcher still
    # runs but the regression is visible. This matches the paper's
    # original Eq. 11 derivation under the implicit "params only on
    # GPU" assumption — strictly an UNDER-estimate for full FT, so the
    # searcher will pick configs that OOM at runtime; the warning
    # signals that the trace cache should be refreshed.
    #
    # Delegated to ``model_state_present_bytes`` so the searcher's
    # inline fast-path peak (``search/exhaustive.py``) and this
    # validator share a single implementation. The two formulas
    # previously diverged silently when only the validator was updated
    # to charge full Adam state per persistent chunk (commit d908bf28),
    # leaving the searcher's pruning optimistic for full FT.
    model_state_present = model_state_present_bytes(cfg, layout, trace)

    # --- Per-block activation policy -----------------------------------
    # NONE / CKPT / SWAP / OFFLOAD blocks contribute differently to the live set:
    #   NONE:    full activation bytes retained from fwd to bwd.
    #   CKPT:    0 bytes retained; bumps peak at first op of this block
    #            (S_chunk + activation_size — recompute materializes both).
    #   SWAP:    0 bytes retained in steady state (see module docstring).
    #   OFFLOAD: full activation bytes retained (same as NONE), AND a
    #            smaller backward-side bump of ``S_chunk`` (chunk gather only,
    #            activations already counted in live_none — see Option B
    #            §4.1). Timed at the LAST forward op of the block, which is
    #            the op-walk index closest to that block's first backward op
    #            (backward processes blocks in reverse forward order; the
    #            forward-only op-walk lands the bump at the symmetrically
    #            closest forward index).
    forward_ops_by_block = _group_ops_by_block(trace)
    tree_index_map = block_tree_index_map(trace)
    cross_attn_bytes = cross_attn_persist_bytes(trace, block_map, tree_index_map)

    # Resolve "first op index" for each CKPT block; used to schedule the
    # checkpoint recomputation bump. If the block has no ops (degenerate
    # test input) the bump lands at op index -1 and is ignored below.
    ckpt_bump_op: dict[int, int] = {}
    # Resolve "last op index" for each OFFLOAD block; used to schedule the
    # backward-window chunk-gather bump (§4.1). The last forward op is the
    # closest forward index to the block's first backward op — backward
    # walks blocks in reverse forward order, so the OFFLOAD-block gather
    # peak materializes at that op-walk position when the forward
    # activations are still resident.
    offload_bump_op: dict[int, int] = {}
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)
        elif mode is BlockMode.OFFLOAD:
            offload_bump_op[op_idxs[-1]] = int(block_id)

    # Retained-activation contribution from NONE + OFFLOAD blocks —
    # constant across the op-walk (these activations are live from their
    # first op through the end of forward). OFFLOAD retains activations
    # symmetrically to NONE; the additional chunk-gather bump fires only
    # at the per-block backward window via ``offload_bump_op``.
    retained_none_bytes = 0
    for block_id_raw, act_sz in trace.activation_sizes.items():
        # ``activation_sizes`` is typed ``dict[BlockId, int]`` but
        # pickled maps may use int keys; normalize.
        bid = BlockId(int(block_id_raw))
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            retained_none_bytes += act_sz
        # CKPT: only live during its recomputation window -> handled
        #       by the per-op bump below.
        # SWAP: live only during the block's forward compute; assumed
        #       to overlap free GPU memory (§3.3).

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
            # Backward-only ops are out of scope for the forward
            # op-walk. Eq. 8-10 explicitly walk forward ops.
            continue

        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)

        # CKPT bump: when we hit the first op of a CKPT block, the
        # recomputation materializes that block's activations *in
        # addition to* any retained activations. This models the peak
        # during the backward-driven recomp window that lines up with
        # this op's forward-equivalent workload.
        ckpt_extra = 0
        if i in ckpt_bump_op:
            ckpt_extra = trace.activation_sizes.get(BlockId(ckpt_bump_op[i]), 0)

        # OFFLOAD backward-gather bump (Option B §4.1): the chunk is
        # re-gathered into the buffer pool for this block's backward
        # while the forward-retained activations are still live. The
        # bump is ``S_chunk`` only (chunk buffer materialization) — the
        # activation bytes are already counted in ``live_none`` because
        # OFFLOAD blocks retain activations like NONE. This is strictly
        # smaller than the CKPT bump (which pays
        # ``S_chunk + activation_size`` because recompute materializes
        # both). Lands at the LAST forward op of the OFFLOAD block —
        # the closest op-walk index to that block's first backward op
        # in the reverse-order backward traversal.
        offload_extra = 0
        if i in offload_bump_op:
            offload_extra = layout.S_chunk

        op_cross_attn = op_cross_attn_surcharge(op, cross_attn_bytes, tree_index_map)

        candidate = (
            model_state_present
            + live_none
            + ckpt_extra
            + offload_extra
            + op_cross_attn
            + intra
            + inter
        )
        if candidate > raw_peak:
            raw_peak = candidate

    # If the trace has no forward ops (degenerate test input) fall back
    # to a static estimate. This keeps the function total.
    if raw_peak == 0:
        raw_peak = model_state_present + retained_none_bytes

    # Ground-truth forward cap from the profiler's hook-less steady pass.
    #
    # Per-block cap (TRACE_VERSION>=6): lightweight block-level hooks during
    # the steady forward record each block's peak bytes. The MAX across
    # those per-block peaks is a strict upper bound on the forward peak
    # regardless of which blocks are NONE/CKPT/SWAP — CKPT and SWAP blocks
    # free their activations before the next block runs, so a mixed
    # configuration's forward peak can never exceed the per-block max
    # observed under the all-NONE profile. CKPT blocks do add a
    # recomputation peak during BACKWARD (one block's activations
    # rematerialized at a time, serially), which isn't captured during
    # this forward-only measurement — add the max single-CKPT-block
    # activation bytes on top.
    #
    # This supersedes the v5 aggregate-only cap (which only applied when
    # n_checkpoint==0 && n_swap==0, making it a no-op for the 7B LoRA
    # test where the searcher picks n_checkpoint≈9). With per-block data
    # the cap tightens ALL configs, including fractional-NONE.
    #
    # Fallback order:
    #   1. Per-block dict populated (v6+) -> use forward_max_block + ckpt_bump
    #   2. Aggregate-only populated (v5, or v6 when discover_blocks failed)
    #      AND all-NONE cfg -> use aggregate
    #   3. Neither -> preserve op-walk raw_peak
    #
    # LAYERING (post-d908bf28 fix): ``raw_peak`` is the sum of two layers,
    #
    #     raw_peak = model_state_present + op_walk_activation_portion
    #
    # where ``op_walk_activation_portion`` is everything the per-op walk
    # contributed on top of the static persistent + buffer-pool footprint
    # (live NONE activations, CKPT/OFFLOAD bumps, cross-attn surcharge,
    # intra/inter deltas).
    #
    # ``measured_cap`` from :func:`hot_iter_peak_cap` is captured during
    # the profiler's hook-less steady FORWARD pass (see
    # ``profiler/trace.py::run_trace`` around the ``steady_fwd_peak_bytes``
    # assignment). At that moment:
    #
    #     - fp16 params are resident (= ``N_chunk * S_chunk`` bytes)
    #     - the forward's max activation is resident
    #     - NO grads, NO fp32 master, NO Adam moments (the optimizer has
    #       not been constructed yet — :class:`OnDemandTensorMgr` only
    #       offloads params, and on-demand mode short-circuits this whole
    #       capture path anyway, so when ``measured_cap is not None`` we
    #       know the FAST path ran with full params on GPU).
    #
    # So the cap decomposes as:
    #
    #     measured_cap = profile_time_model_state + measured_activation_cap
    #     profile_time_model_state = N_chunk * S_chunk
    #
    # Applying the cap to the FULL ``raw_peak`` (the pre-fix code) was a
    # bug: it silently erased the per-chunk Adam-state contribution that
    # commit d908bf28 added via :func:`model_state_present_bytes`. For a
    # full-FT trace where ``persistent_factor ~= 8.0``, the cap clamped
    # raw_peak to ~``measured_activation_cap + N_chunk * S_chunk`` while
    # the true minimum is ``model_state_present + measured_activation_cap``
    # — Codex confirmed with a synthetic full-FT trace that the
    # pre-fix peak (~148 MB) was less than the model-state lower bound
    # alone (~2.36 GB).
    #
    # The fix bounds ONLY the activation portion, leaving the model-state
    # term intact through the cap. Layered application is centralized in
    # :func:`apply_hot_iter_cap` so this site and the searcher's inline
    # fast path (search/exhaustive.py) cannot drift.
    measured_cap = hot_iter_peak_cap(trace, block_map, cfg, layout)
    raw_peak = apply_hot_iter_cap(raw_peak, model_state_present, measured_cap, layout)

    scaled = int(ALPHA_FRAGMENTATION * raw_peak)
    LOG.debug(
        "estimate_peak: n_persist=%d n_buffer=%d n_swap=%d n_ckpt=%d n_offload=%d "
        "raw=%dB alpha=%.2f -> %dB",
        cfg.n_persist,
        cfg.n_buffer,
        cfg.n_swap,
        cfg.n_checkpoint,
        cfg.n_offload,
        raw_peak,
        ALPHA_FRAGMENTATION,
        scaled,
    )
    return scaled


__all__ = [
    "ALPHA_FRAGMENTATION",
    "_saved_tensor_bytes_per_block",
    "block_tree_index_map",
    "cross_attn_persist_bytes",
    "estimate_cpu_footprint",
    "estimate_peak",
    "hot_iter_peak_cap",
    "model_state_present_bytes",
    "op_cross_attn_surcharge",
]
