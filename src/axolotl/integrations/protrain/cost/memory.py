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
    if module_path.startswith("encoder.") or module_path == "encoder":
        return 0
    if module_path.startswith("decoder.") or module_path == "decoder":
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
        return dict(persisted)
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
        ckpt_recomp_bump = 0
        has_offload = False
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            mode = block_map.get(bid, BlockMode.NONE)
            if mode is BlockMode.CKPT:
                if act_sz > ckpt_recomp_bump:
                    ckpt_recomp_bump = act_sz
            elif mode is BlockMode.OFFLOAD:
                has_offload = True
        offload_bump = layout.S_chunk if (has_offload and layout is not None) else 0
        return forward_max_block_peak + ckpt_recomp_bump + offload_bump
    if (
        trace.steady_fwd_peak_bytes > 0
        and cfg is not None
        and cfg.n_checkpoint == 0
        and cfg.n_swap == 0
        and cfg.n_offload == 0
    ):
        return trace.steady_fwd_peak_bytes
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
    by up to K× but guarantees any saved tensor fits any slot — see
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
    non_persist = max(0, layout.N_chunk - cfg.n_persist)
    # Under sharding each rank holds 1/gpu_count of each chunk. Ceiling
    # division is applied PER CHUNK so small chunks don't underreport
    # when ``S_chunk`` isn't divisible by ``gpu_count`` — summing
    # ``total_bytes`` first and dividing once would round only at the
    # aggregate, undercounting the trailing-rank padding by up to
    # ``non_persist - 1`` bytes.
    per_rank_divisor = hw.gpu_count if hw.zero3_shard else 1
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
    """Resident model-state bytes for ``(n_persist, n_buffer)`` chunks.

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
    n_persist = max(0, min(cfg.n_persist, layout.N_chunk))
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist))

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
        LOG.warning(
            "model_state_present_bytes: trace.model_state_bytes is missing "
            "or zero (model_state_bytes=%d, fp16_total=%dB); falling back "
            "to the legacy n_persist*S_chunk multiplier. The peak estimate "
            "will UNDER-count full optimizer state — refresh the profiler "
            "trace cache (TRACE_VERSION bump) to restore Eq. 11 fidelity.",
            model_state_total,
            fp16_total_bytes,
        )
        persistent_factor = 1.0
    # Buffer slot during backward = fp16 params (gathered) + fp16 grads
    # (accumulated). 2.0 is a strict upper bound on the buffer pool's
    # transient peak; the optimizer state never lives in the buffer
    # pool itself (non-persistent chunks have their fp32 master + m +
    # v on CPU and only stream onto GPU through the optimizer's own
    # transient buffers, accounted for separately by the runtime model).
    buffer_factor = 2.0
    return int(
        n_persist * layout.S_chunk * persistent_factor
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
    "block_tree_index_map",
    "cross_attn_persist_bytes",
    "estimate_cpu_footprint",
    "estimate_peak",
    "hot_iter_peak_cap",
    "model_state_present_bytes",
    "op_cross_attn_surcharge",
]
