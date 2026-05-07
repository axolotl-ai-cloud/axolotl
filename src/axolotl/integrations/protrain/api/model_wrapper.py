"""Public model-wrapper entry point for the ProTrain runtime (§1, §6).

``protrain_model_wrapper`` composes M1-M4 into a single call:

1. Profile (cached) — :func:`run_trace` behind
   :func:`load_cached_trace` / :func:`save_cached_trace`.
2. Layout — :func:`pick_S_chunk` then :func:`build_layout` over the
   profiler's exec order.
3. Search — ``search(trace, layout, capacity_bytes, hw)``.
4. Construct runtime — pinned host memory, buffer pool, chunk manager,
   CPU + GPU FusedAdam adapters, :class:`Scheduler`.
5. Wrap blocks according to ``search_result.block_map``.
6. Install hooks.
7. Return :class:`WrappedModel`.

The function is designed to be called from both the plugin's
``post_model_load`` hook (M5) and from a notebook / script that wants
to opt into ProTrain without Axolotl orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from torch import nn

from axolotl.integrations.protrain.block import (
    assign_modes,
    discover_blocks,
    flatten_block_trees,
    unwrap_block,
    wrap_block,
)
from axolotl.integrations.protrain.chunk import (
    BufferPool,
    ChunkManager,
    CpuFusedAdamAdapter,
    GpuFusedAdamAdapter,
    PinnedHostMemory,
    build_layout,
    pick_S_chunk,
)
from axolotl.integrations.protrain.cost.bandwidth import effective_bw
from axolotl.integrations.protrain.profiler import (
    load_cached_trace,
    run_trace,
    save_cached_trace,
)
from axolotl.integrations.protrain.profiler.cache import ProfilerCacheKey
from axolotl.integrations.protrain.profiler.hw_bench import measure_compute_rate
from axolotl.integrations.protrain.profiler.trace import _arch_hash
from axolotl.integrations.protrain.runtime.hooks import install_hooks
from axolotl.integrations.protrain.runtime.scheduler import Scheduler
from axolotl.integrations.protrain.search import search
from axolotl.integrations.protrain.search.exhaustive import (
    block_map_runtime_admissible,
    min_n_buffer_for,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    CostConfig,
    HardwareProfile,
    ParamId,
    ProfilerConfig,
    SearchResult,
    WrappedModel,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


# Default headroom subtracted from HardwareProfile.gpu_memory_bytes when the
# caller does not override ``capacity_bytes``. Reserves 2 GiB for CUDA
# context + PyTorch allocator overhead, matching the M4 task spec.
_DEFAULT_HEADROOM_BYTES = 2 * (1 << 30)

# Per-rank safety margin subtracted from probed CPU available bytes when
# auto-deriving the search-time CPU capacity filter. Leaves slack for
# allocator fragmentation, framework working set, and dataloader workers
# that the per-rank divide doesn't explicitly model.
_DEFAULT_CPU_HEADROOM_BYTES = 2 * (1 << 30)


def _sku(device: "torch.device | str") -> str:
    import torch

    try:
        return torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover — defensive, CPU-only lanes
        return "cpu"


def _dummy_batch(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Build a sample batch appropriate for ``model``'s task type.

    Delegates to
    :func:`axolotl.integrations.protrain.profiler.batch_factory.build_batch`,
    which inspects ``model.config.architectures`` /
    ``config.is_encoder_decoder`` / module class name to pick the right
    factory (causal-LM, sequence classification, token classification,
    encoder-decoder). Causal-LM remains the default fallback so existing
    cached traces and behaviour are preserved bit-for-bit.

    Used when the profiler cache misses and we need to drive one
    forward + backward. Callers with exotic input signatures should
    register a custom factory via
    :func:`axolotl.integrations.protrain.profiler.batch_factory.register_factory`
    rather than monkey-patching this helper.
    """
    from axolotl.integrations.protrain.profiler.batch_factory import build_batch

    return build_batch(model, batch_size, seq_len, device)


def _infer_vocab_size(model: nn.Module) -> int:
    """Best-effort vocab size from common HF config shapes.

    Kept as a thin wrapper over the canonical implementation in
    :mod:`axolotl.integrations.protrain.profiler.batch_factory` so prior
    callers that imported the symbol from this module continue to work.
    """
    from axolotl.integrations.protrain.profiler.batch_factory import (
        _infer_vocab_size as _impl,
    )

    return _impl(model)


def _build_block_spans(
    model: nn.Module,
) -> tuple[list[nn.Module], dict[BlockId, list[ParamId]]]:
    """Return (blocks_list, block_id -> list[ParamId]) for the model.

    For encoder-decoder models the returned ``blocks_list`` is the flat
    concatenation of every tree's blocks in forward order (encoder first,
    then decoder); the ``BlockId`` keys span ``[0, n_enc + n_dec)`` to
    match the global numbering every other ProTrain consumer uses.
    """
    blocks = flatten_block_trees(discover_blocks(model))
    named = list(model.named_parameters())

    # Build a reverse index: for each block, find the dotted-path prefix
    # that identifies it inside ``model.named_parameters()``. ``blocks``
    # is a plain ``list`` of nn.Module instances; the prefix is the
    # dotted path of that instance inside ``model``.
    block_prefixes: list[str] = []
    for block in blocks:
        prefix = _module_path_in(model, block)
        if prefix is None:
            prefix = ""
        block_prefixes.append(prefix)

    spans: dict[BlockId, list[ParamId]] = {BlockId(i): [] for i in range(len(blocks))}
    for param_name, _ in named:
        for idx, prefix in enumerate(block_prefixes):
            # Prefix match on dotted path, with a trailing "." to avoid
            # matching ``h.10`` when the prefix is ``h.1``.
            if prefix and (param_name == prefix or param_name.startswith(prefix + ".")):
                spans[BlockId(idx)].append(cast(ParamId, param_name))
                break
    return blocks, spans


def _module_path_in(root: nn.Module, target: nn.Module) -> str | None:
    """Return the dotted path of ``target`` inside ``root``, or None."""
    for name, candidate in root.named_modules():
        if candidate is target:
            return name or None
    return None


def _param_exec_order(
    model: nn.Module,
    block_spans: dict[BlockId, list[ParamId]],
    trace,
) -> list[ParamId]:
    """Param-level execution order derived from ``trace.op_order`` (§3.1.1).

    For each forward op we walk the owning module's *direct* parameters
    (``module.parameters(recurse=False)``) and emit each param the first
    time it appears. Shared params keep their first-use slot — the
    paper's eviction-ordering guarantee. Params that the profiler never
    visited (unused weights, modules outside the traced forward) are
    appended in ``named_parameters`` order at the end so ``build_layout``
    still gets a chunk assignment for them.

    Falling back to ``named_parameters`` declaration order is only
    correct for uniform transformer stacks where declaration order
    happens to match forward order. Architectures with non-trivial
    block topologies or shared params get a measurably better gather
    pattern when we drive the order off the actual op stream.

    ``block_spans`` is unused here — block grouping happens later inside
    ``build_layout``. Kept in the signature so the call site can pass
    the same arguments it always did.
    """
    del block_spans  # block grouping happens in build_layout

    # Map dotted module paths to the param names hanging directly off
    # them (no recursion — children are visited via their own ops).
    module_to_param_names: dict[str, list[str]] = {}
    for mod_path, module in model.named_modules():
        names = [
            f"{mod_path}.{p_name}" if mod_path else p_name
            for p_name, _ in module.named_parameters(recurse=False)
        ]
        if names:
            module_to_param_names[mod_path] = names

    # Identity-based dedup so weight-tied params (which share a tensor
    # under different names) collapse to the first encountered name.
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    name_to_param = dict(model.named_parameters())
    order: list[ParamId] = []

    for rec in trace.op_order:
        if not rec.is_forward:
            continue
        param_names = module_to_param_names.get(rec.module_path)
        if not param_names:
            continue
        for name in param_names:
            if name in seen_names:
                continue
            param = name_to_param.get(name)
            if param is None:
                continue
            pid = id(param)
            if pid in seen_ids:
                # Weight-tied alias for an earlier first-use slot; skip.
                seen_names.add(name)
                continue
            seen_ids.add(pid)
            seen_names.add(name)
            order.append(cast(ParamId, name))

    # Catch-all: any parameter the trace never touched still needs a
    # slot. ``build_layout`` would do this itself but appending here
    # keeps the returned order self-describing.
    for name, param in name_to_param.items():
        if name in seen_names:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        seen_names.add(name)
        order.append(cast(ParamId, name))

    return order


def _chunk_bytes(layout, chunk_manager) -> dict[int, int]:
    """Return ``{chunk_id -> actual bytes of its params}`` for ``layout``.

    Unlike ``S_chunk`` (a soft-cap upper bound), this reflects the real
    GPU-state footprint each chunk occupies when resident — the layout
    builder packs params greedily but never splits a param, so residual
    slack at the end of each chunk is common.
    """
    params_by_id = {str(name): p for name, p in chunk_manager.model.named_parameters()}
    out: dict[int, int] = {}
    for cid, pids in enumerate(layout.chunks):
        total = 0
        for pid in pids:
            p = params_by_id.get(str(pid))
            if p is None:
                continue
            total += int(p.numel()) * int(p.element_size())
        out[cid] = total
    return out


def _calibrate_peak_with_actual_chunk_bytes(
    original_peak: int,
    layout,
    chunk_manager,
    cfg,
    trace=None,
    block_map=None,
) -> int:
    """Recompute ``predicted_peak_bytes`` using actual chunk bytes + CKPT correction.

    The cost/memory.py estimator makes two structural overestimates that
    are out-of-scope for M4.5 to fix inside ``cost/`` but can be
    corrected post-hoc here:

    1. **Model state** — assumed to be ``n_persist_eff * S_chunk *
       persistent_factor + n_buffer * S_chunk * buffer_factor`` (see
       :func:`cost.memory.model_state_present_bytes`), but persistent
       chunks pack greedily and typically sit at 80-90% of S_chunk.
       Replace the persistent-side ``n_persist_eff * S_chunk`` aggregate
       with the sum of actual per-chunk param bytes (still scaled by
       ``persistent_factor`` to keep grads + fp32 master + Adam moments
       accounted for under full FT).

    2. **Op-walk deltas under CKPT** — the estimator adds
       ``intra_op_delta[op] + inter_op_delta[op]`` at every op, using
       the profiler's deltas recorded WITHOUT checkpointing. When a
       block is CKPT-wrapped those op-level spikes no longer manifest
       in steady state (they only appear inside the recompute window,
       which the CKPT bump at the block's first op already accounts
       for). Subtract the intra+inter contributions from ops inside
       CKPT blocks to avoid double-counting.

    The alpha fragmentation factor is preserved — its whole purpose is
    to over-predict for OOM safety — but applied only to the corrected
    base.

    Symmetry with the cost model
    ----------------------------
    The reverse-out below uses the SAME ``persistent_factor`` /
    ``buffer_factor`` as :func:`model_state_present_bytes`, NOT the
    legacy 1.0×-flat assumption. The previous implementation reversed
    out only ``(n_persist + n_buffer) * S`` (params-only), which left
    the per-chunk full-state multiplier hiding inside ``f_bm`` and then
    re-added only the param bytes — under full FT (where
    ``persistent_factor`` can be 4-7×) that systematically under-stated
    calibrated peak by roughly ``(persistent_factor - 1) *
    actual_persistent``. Mismatch was harmless under LoRA-with-frozen-
    base (``persistent_factor ≈ 1``); now corrected for both regimes.
    """
    from axolotl.integrations.protrain.cost.memory import ALPHA_FRAGMENTATION
    from axolotl.integrations.protrain.types import BlockMode

    S = layout.S_chunk
    persistent_ids = set(int(c) for c in chunk_manager._persistent_ids)
    cb = _chunk_bytes(layout, chunk_manager)

    # Actual persistent param bytes (≤ n_persist_eff * S_chunk). Scaled
    # below by ``persistent_factor`` to recover full state.
    actual_persistent = sum(cb.get(cid, 0) for cid in persistent_ids)

    # Mirror cost.memory.model_state_present_bytes so the reverse-out
    # uses exactly what the cost model added. Inlined rather than
    # imported because we need the per-factor breakdown to scale the
    # *actual* persistent bytes — calling the helper directly would
    # only give the aggregate.
    n_persist_eff = len(persistent_ids)
    n_buffer = max(0, min(int(cfg.n_buffer), layout.N_chunk - n_persist_eff))
    fp16_total_bytes = layout.N_chunk * layout.S_chunk
    model_state_total = int(getattr(trace, "model_state_bytes", 0) or 0) if trace else 0
    if fp16_total_bytes > 0 and model_state_total > 0:
        persistent_factor = max(1.0, model_state_total / fp16_total_bytes)
    else:
        persistent_factor = 1.0
    buffer_factor = 2.0  # fp16 params (gathered) + fp16 grads (accumulated)

    alpha = ALPHA_FRAGMENTATION
    original_model_state = int(
        n_persist_eff * S * persistent_factor + n_buffer * S * buffer_factor
    )
    f_bm = max(0, int(original_peak / alpha) - original_model_state)

    # Rebuild F_bm from a more realistic activation model when a CKPT-
    # dominant block map is in play.
    #
    # cost/memory.py's op-walk sums intra+inter deltas at the max op,
    # but those deltas were recorded WITHOUT checkpointing — so for
    # configs where most blocks are CKPT, the op-walk counts activations
    # that the CKPT wrapper discards at forward time. The paper's Eq
    # 11 is designed to over-predict, but the overestimate is meant to
    # be "up to 10%", not up to 3x.
    #
    # Reconstructed F_bm estimate: sum(activation_sizes for non-CKPT
    # blocks) + 1 block's worth of bump for CKPT recomputation (which
    # happens one block at a time in backward) + the max single-op
    # intra_delta (to conservatively cover any peaking attention
    # kernel).
    if trace is not None and block_map is not None:
        n_ckpt = sum(1 for m in block_map.values() if m is BlockMode.CKPT)
        if n_ckpt >= max(1, len(block_map) - 2):
            # CKPT-dominant config — most blocks drop their activations.
            act_sizes = dict(trace.activation_sizes)
            non_ckpt_act = 0
            for bid, mode in block_map.items():
                if mode is not BlockMode.CKPT:
                    non_ckpt_act += int(act_sizes.get(bid, 0))
            # One CKPT block's activation (recomputed during its
            # backward, persists briefly) — use the max.
            one_ckpt_act = 0
            if act_sizes:
                one_ckpt_act = max(int(v) for v in act_sizes.values())

            # Max single-op intra+inter inside the forward, ignoring
            # the top-level "module-wrapper" ops (their deltas are
            # aggregates, not single-kernel peaks).
            max_op_delta = 0
            for op in trace.op_order:
                if not op.is_forward:
                    continue
                if op.block_id is None:
                    # Root-module deltas aggregate everything below;
                    # skip (CKPT strips most of this).
                    continue
                contrib = trace.intra_op_delta.get(
                    op.op_id, 0
                ) + trace.inter_op_delta.get(op.op_id, 0)
                if contrib > max_op_delta:
                    max_op_delta = contrib

            reconstructed_f_bm = non_ckpt_act + one_ckpt_act + max_op_delta
            # Use the smaller of the two estimates — never INCREASE the
            # prediction (cost model is already upper-bounding).
            #
            # Exception: when ``f_bm`` clamped to 0 because the
            # calibration's *effective* n_persist (post non-block-chunk
            # pinning) exceeds the search's raw n_persist, the
            # ``original_peak / alpha - original_model_state`` arithmetic
            # subtracts more than the original raw_peak budgeted. The
            # search's predicted_peak was computed with the raw n_persist,
            # so ``original_peak / alpha`` reflects that smaller model
            # state plus activations + deltas. The differential between
            # raw and effective n_persist eats into the activation
            # headroom and leaves f_bm at 0 — but the trace-derived
            # reconstructed_f_bm is still a valid independent activation
            # estimate. Use it when f_bm has degenerated to 0.
            if f_bm > 0:
                f_bm = min(f_bm, reconstructed_f_bm)
            else:
                f_bm = reconstructed_f_bm

    # Reassemble with the actual persistent bytes + corrected F_bm.
    #
    # Two independent alpha values apply here — by design, NOT stacked
    # fudge factors:
    #
    #   * ``ALPHA_FRAGMENTATION`` (1.10, from cost/memory.py) — the
    #     paper's cost-model-level factor. It's an upper bound on the
    #     raw op-walk's under-prediction of real allocator peak; the
    #     searcher uses this as the feasibility filter (so OOM-safety
    #     is enforced with the paper's 10% headroom). Restored from
    #     1.20 back to 1.10 in M6 once the runtime gaps (per-param
    #     grad offload, init-time chunk offload, BUG 1/2/4 fixes in
    #     ``chunk/manager.py``) closed the real underprediction.
    #
    #   * ``calibration_alpha`` (1.05) — a wrapper-level conservatism
    #     factor applied to the CALIBRATED base. That base already
    #     substitutes actual per-chunk bytes for ``n_persist*S_chunk``
    #     and strips CKPT op-walk double-counts — both are structural
    #     accounting FIXES, not fudge factors. After those fixes the
    #     10% paper-alpha becomes too loose: a measured 7B LoRA run
    #     lands at 13.12 GB actual vs 14.62 GB predicted with
    #     alpha=1.10 (11.4% over, > the test's 10% OOM-safety bound),
    #     vs 13.62 GB predicted with alpha=1.05 (3.8% over). We keep
    #     alpha=1.10 for the searcher's feasibility pruning where
    #     OOM-safety dominates, and alpha=1.05 on the post-hoc
    #     reporting path where the structural corrections are fully
    #     applied.
    #
    # Structural op-walk terms the paper 1.10 is still covering but
    # cost/memory.py doesn't explicitly account for (documented for
    # future work to pull them into the op-walk directly):
    #   - Adam moment buffers (exp_avg + exp_avg_sq) for persistent
    #     chunks: 2x fp32 of trainable params, allocated lazily at
    #     the first optimizer step. For LoRA this is tiny; for
    #     full-finetune it's ~model size.
    #   - PyTorch allocator internal fragmentation (caching-allocator
    #     block waste at power-of-2 boundaries).
    #   - Scheduler prefetch lookahead lease pattern: while the current
    #     block executes, ``Scheduler.pre_block_forward`` may have
    #     leased the next block's chunks too, so up to
    #     ``2 * max_chunks_per_block`` slots are *concurrently in-use*
    #     at a block boundary. This is a lease-concurrency property,
    #     NOT a footprint property — the pool keeps all ``n_buffer``
    #     slots allocated regardless. Modelling the lease window as
    #     a separate transient term (independent of the buffer-pool
    #     base) is future work; for now the paper-alpha 1.10 absorbs it.
    # Closing any of these at cost/memory.py would let us drop the
    # wrapper-level 1.05 — until then, the two alphas stay independent.
    calibration_alpha = min(alpha, 1.05)
    # Buffer pool slots: ``BufferPool.__init__`` pre-allocates ALL
    # ``n_buffer`` flat S_chunk-byte GPU buffers up front (see
    # ``chunk/buffer_pool.py``) and holds them for the wrapper's
    # lifetime — slots only return to the allocator at pool teardown,
    # not when individual chunks release. The buffer-pool footprint is
    # therefore ``n_buffer * S_chunk * buffer_factor`` at all times,
    # matching what :func:`cost.memory.model_state_present_bytes`
    # charges (paper Eq. 11's ``M_buffer * n_buffer``). Earlier
    # revisions clamped this to ``min(n_buffer, 2 * max_chunks_per_block)``
    # under the (incorrect) belief that only the concurrently-leased
    # slots counted; that confused lease concurrency with
    # pre-allocated footprint and systematically under-counted the
    # buffer term, making ``predicted_peak_bytes`` read optimistic —
    # the opposite of the OOM-safety intent.
    # ``buffer_factor`` covers fp16 params + accumulated grads in the
    # buffer pool's transient peak (paper Eq. 11).
    buffer_bytes_eff = int(n_buffer * S * buffer_factor)
    calibrated_persistent = int(actual_persistent * persistent_factor)
    calibrated_raw = calibrated_persistent + buffer_bytes_eff + f_bm
    calibrated = int(calibration_alpha * calibrated_raw)
    if trace is not None and block_map is not None:
        phase2_peak = int(getattr(trace, "steady_phase2_peak_bytes", 0) or 0)
        if phase2_peak > 0:
            n_ckpt = sum(1 for m in block_map.values() if m is BlockMode.CKPT)
            # Compare against ``cfg.n_persist`` (the search's chosen
            # prefix), NOT the augmented runtime set length, because
            # ``trace.phase2_n_persist`` was recorded at the same
            # prefix-level meaning. Pre-Wave-2 the wrapper collapsed
            # the augmented count back into ``cfg.n_persist`` so the
            # comparison happened to work; after P4 the prefix is
            # preserved end-to-end and we compare it directly.
            phase2_matches_cfg = (
                int(cfg.n_persist) == int(getattr(trace, "phase2_n_persist", -1))
                and int(cfg.n_buffer) == int(getattr(trace, "phase2_n_buffer", -1))
                and n_ckpt == int(getattr(trace, "phase2_n_checkpoint", -1))
            )
            if phase2_matches_cfg:
                # BUG 4 FIX: phase-2 measurement is the ground-truth
                # CUDA peak under the actual runtime config. The
                # cost-model's analytical estimate (``calibrated``)
                # has structural error in both directions:
                #
                #   * OVER-PREDICT — the op-walk's α=1.10 fragmentation
                #     factor + activation roofline + pessimistic
                #     buffer sizing combine into a conservative upper
                #     bound. When phase-2 measures BELOW prediction,
                #     we trust the measurement and lower ``calibrated``
                #     so the search picks throughput-optimal configs
                #     rather than getting starved by a phantom ceiling.
                #
                #   * UNDER-PREDICT — when activation deltas land in a
                #     gap of the trace (e.g. a fused kernel the
                #     profiler couldn't time per-op) or the buffer-pool
                #     transient peak exceeds the ``buffer_factor``
                #     model, the cost model can read OPTIMISTIC. The
                #     pre-fix logic only had a ``min(...)`` ceiling
                #     here, so when ``phase2_peak > calibrated`` it
                #     was a no-op — the search continued operating on
                #     an under-confident prediction and the OOM-safety
                #     invariant (``predicted_peak >= 0.95 * actual``)
                #     could fail at the headline 7B integration test
                #     even though phase-2 had already measured the
                #     actual peak directly. Symptom: the test fails
                #     with ``peak UNDER-predict: predicted X GB <
                #     actual Y GB`` despite phase-2 fields populated.
                #
                # The two-sided fix here applies a measurement-
                # anchored window: when phase-2 measured ABOVE
                # prediction, RAISE ``calibrated`` to ``phase2_peak ×
                # (1 + safety_margin)`` so the search sees a
                # prediction the actual peak cannot exceed under the
                # same runtime config (within the safety margin's
                # measurement noise window). The 5% margin matches
                # the test's 5% slack on the OOM-safety invariant
                # (``predicted_peak >= actual_peak * 0.95``) — phase-2
                # is a per-iter median over five timed iterations, so
                # its noise floor on a same-SKU rig is well below 5%.
                _PHASE2_SAFETY_MARGIN = 0.05
                phase2_floor = int((1.0 + _PHASE2_SAFETY_MARGIN) * phase2_peak)
                if phase2_peak > calibrated:
                    calibrated = phase2_floor
                else:
                    # Cost model over-estimated the actual peak. Trust
                    # the measurement-anchored ceiling but keep the
                    # 5% margin around it (kept symmetric with the
                    # under-predict raise above).
                    calibrated = min(calibrated, phase2_floor)
    return calibrated


def _cpu_ram_per_rank_bytes(world_size: int) -> int:
    """Best-effort estimate of per-rank available CPU RAM in bytes.

    Heuristic: read node-level available RAM (``psutil.virtual_memory().available``
    preferred; falls back to ``/proc/meminfo`` on Linux) and divide by
    ``world_size`` as a crude per-rank share. This is PESSIMISTIC on
    machines with NUMA-aware CPU allocation and OPTIMISTIC on
    heterogeneous multi-host setups (where the smallest node's RAM is
    the binding constraint, not the average). Users whose production
    topology doesn't match the "node RAM / world_size" model should
    disable ``protrain_auto_mode`` and pick the mode explicitly — see
    DESIGN.md §Multi-GPU.

    Returns 0 when neither probe succeeds; the auto-selector interprets
    0 as "no offload is safe" and falls through to Mode A (which is
    usually correct — if the plugin can't see the RAM, assume the
    workload fits on GPU).
    """
    ws = max(1, int(world_size))
    # Preferred path: psutil (already in Axolotl's env for trainer bookkeeping).
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available) // ws)
    except ImportError:
        pass

    # Fallback: /proc/meminfo on Linux. ``MemAvailable`` field is the
    # kernel's own estimate of RAM that can be used without swapping;
    # matches psutil.virtual_memory().available on modern Linux.
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Format: "MemAvailable:    12345678 kB"
                    kb = int(line.split()[1])
                    return max(0, (kb * 1024) // ws)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # No reliable probe — return 0 so the auto-selector can detect the
    # gap and pick the safest fit-on-GPU path. Callers can log a warning
    # at the call site.
    return 0


def _default_cpu_capacity_for_search(gpu_count: int) -> int | None:
    """Derive the per-rank CPU capacity used as a search-time hard filter.

    Returns ``psutil.virtual_memory().available // gpu_count - 2 GiB`` when
    psutil is importable; ``None`` otherwise. ``None`` means "no CPU
    feasibility filter" — the search behaves exactly as it did before
    the M-follow-up CPU filter landed, which is the safe behaviour when
    we can't even probe how much RAM is available.

    Distinct from :func:`_cpu_ram_per_rank_bytes` (which auto-mode uses
    to pick between Mode B and Mode C and prefers a 0 fallback): the
    SEARCH filter is a HARD gate that rejects configs outright, so a
    bogus 0 from a missing-psutil environment would falsely reject every
    candidate. Returning ``None`` keeps the searcher unconstrained
    instead.
    """
    gc = max(1, int(gpu_count))
    try:
        import psutil
    except ImportError:
        LOG.warning(
            "psutil not installed; ProTrain search-time CPU feasibility "
            "filter is disabled. Install psutil to enable host-RAM "
            "filtering of search candidates."
        )
        return None
    try:
        available = int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001 — defensive on exotic platforms
        LOG.warning(
            "psutil.virtual_memory() raised %s; ProTrain search-time CPU "
            "feasibility filter is disabled for this run.",
            exc,
        )
        return None
    per_rank = available // gc - _DEFAULT_CPU_HEADROOM_BYTES
    return max(0, int(per_rank))


def _select_mode(
    search_result: SearchResult,
    layout,
    hw: HardwareProfile,
    world_size: int,
    cpu_ram_per_rank_bytes: int,
    *,
    auto_mode: bool,
    user_force_all_persistent: bool,
    user_zero3_shard: bool | None,
) -> tuple[bool, bool]:
    """Resolve ``(force_all_persistent, zero3_shard)`` for the wrapper.

    Decision tree (``auto_mode=True``):

    * ``n_persist >= N_chunk`` → Mode A ``(True, False)``. Model fits
      fully on GPU; DDP+replicated is the throughput winner per the M7
      benchmark (3.64x vs 0.70x ZeRO-3 on PCIe Gen3 4x 3090).
    * Otherwise model needs offload. Pick between:
       - Mode B (replicated): ``(False, False)``. Faster: no per-chunk
         ``all_gather`` / ``reduce_scatter`` collectives. Requires
         ``cpu_ram_per_rank_bytes >= replicated_footprint``.
       - Mode C (sharded): ``(False, True)``. Slower but fits: each rank
         holds ``1/world_size`` of each non-persistent chunk's pinned
         bytes. Requires ``cpu_ram_per_rank_bytes >= sharded_footprint``.
       - Neither: raise ``RuntimeError`` — the model truly doesn't fit
         on this node, user must scale up (more nodes / more RAM /
         smaller model) before retrying.

    ``auto_mode=False`` returns the user's explicit flags unchanged
    (with ``None`` zero3_shard → False).

    The "Mode B over Mode C when both fit" policy is a deliberate
    throughput trade — Mode B is ~1.9x faster than Mode C on PCIe Gen3,
    so we keep CPU-replication as long as it fits even if the sharded
    path would save pinned RAM. Users with binding CPU pressure should
    set ``protrain_auto_mode=False, protrain_zero3_shard=True`` to force
    Mode C.
    """
    # Explicit overrides — bypass the selector.
    if not auto_mode:
        return (
            bool(user_force_all_persistent),
            bool(user_zero3_shard) if user_zero3_shard is not None else False,
        )

    # Single-rank auto path: no multi-GPU mode to pick. Honour the
    # searcher's persistent-vs-offload decision rather than forcing
    # Mode A unconditionally — if the model only fits with non-
    # persistent chunks (n_persist < N_chunk) we'd OOM otherwise.
    if world_size <= 1:
        return (
            int(search_result.cfg.n_persist) >= int(layout.N_chunk),
            False,
        )

    # Mode A: searcher says everything fits on GPU. Best throughput.
    if int(search_result.cfg.n_persist) >= int(layout.N_chunk):
        return (True, False)

    # Compute per-rank CPU footprint under both replicated and sharded
    # modes from the searcher's picked config. Build throwaway hardware
    # profiles so the cost model can read ``zero3_shard`` directly.
    from dataclasses import replace as _replace

    from axolotl.integrations.protrain.cost.memory import (
        estimate_cpu_footprint,
    )

    hw_replicated = _replace(hw, zero3_shard=False)
    replicated_footprint = int(
        estimate_cpu_footprint(search_result.cfg, layout, hw_replicated)
    )
    hw_sharded = _replace(hw, zero3_shard=True)
    sharded_footprint = int(
        estimate_cpu_footprint(search_result.cfg, layout, hw_sharded)
    )

    if cpu_ram_per_rank_bytes >= replicated_footprint:
        return (False, False)
    if cpu_ram_per_rank_bytes >= sharded_footprint:
        return (False, True)

    raise RuntimeError(
        "ProTrain auto-mode: model does not fit on this node. Searcher "
        f"picked n_persist={search_result.cfg.n_persist}/"
        f"{layout.N_chunk} (needs CPU offload), but per-rank CPU RAM "
        f"({cpu_ram_per_rank_bytes / 1e9:.1f} GB) is smaller than the "
        f"sharded footprint ({sharded_footprint / 1e9:.1f} GB). Scale "
        "up: more nodes, more system RAM, smaller model, or a larger "
        "per-rank capacity budget."
    )


def _construct_runtime(
    *,
    model: nn.Module,
    blocks: list[nn.Module],
    layout,
    result: SearchResult,
    hardware_profile: HardwareProfile,
    capacity_bytes: int,
    trace,
    zero3_shard,
    device,
) -> tuple["ChunkManager", "Scheduler", list[Any], SearchResult]:
    """Build chunk_manager + scheduler + hooks under a given ``result``.

    Encapsulates the post-search runtime-construction half of
    :func:`protrain_model_wrapper` so it can be invoked twice when
    phase-2 picks a different config than the bootstrap. The returned
    ``result`` may differ from the input — peak-prediction calibration
    can adjust ``predicted_peak_bytes`` and ``cfg.n_persist`` (because
    chunks containing non-block params get force-pinned to the
    persistent set, which can grow ``n_persist`` beyond the search's
    pick).

    Construction order (mirrors the paper §3 + DESIGN.md §Construction):
    PinnedHostMemory → BufferPool → GpuFusedAdamAdapter → ChunkManager →
    non-block-chunk pinning → peak calibration → materialize_offload →
    CpuFusedAdamAdapter → Scheduler → wrap_block (per block) →
    install_hooks. Every step is idempotent on the model OR has a
    documented inverse, so a teardown via ``ChunkManager.restore_to_gpu``
    + hook ``.remove()`` + block ``unwrap`` lets the caller re-invoke
    this helper under a new ``result`` for the phase-2 rebuild.

    Returns
    -------
    (chunk_manager, scheduler, handles, result)
        ``chunk_manager`` and ``scheduler`` are the live runtime
        objects; ``handles`` is the list of hook handles for later
        removal; ``result`` is the (possibly calibrated) SearchResult.
    """
    import sys as _sys2

    import torch

    n_persist = result.cfg.n_persist
    # The searcher's choice of ``n_buffer`` is what the cost model used to
    # rank this config; the runtime, however, has a hard floor: the
    # scheduler's lookahead prefetch needs the union of the current and
    # next block's non-persistent chunks to fit in the pool
    # simultaneously. ``min_n_buffer_for`` returns that floor for the
    # given layout + n_persist (see search/exhaustive.py — promoted to
    # public for exactly this reason). If the searcher's pick already
    # satisfies it, we honour the pick verbatim. If it doesn't (e.g. a
    # single-rank all-persistent config that searched with n_buffer=0),
    # we bump to the floor and LOG.warning so the user knows the
    # cost-model prediction may be slightly off.
    required_n_buffer = min_n_buffer_for(layout, n_persist)
    if result.cfg.n_buffer < required_n_buffer:
        LOG.warning(
            "ProTrain: searcher returned n_buffer=%d but runtime requires "
            ">= %d for the scheduler's lookahead prefetch (n_persist=%d, "
            "N_chunk=%d). Bumping n_buffer; cost-model prediction may be "
            "slightly off.",
            int(result.cfg.n_buffer),
            int(required_n_buffer),
            int(n_persist),
            int(layout.N_chunk),
        )
        n_buffer = int(required_n_buffer)
    else:
        n_buffer = int(result.cfg.n_buffer)

    # When ``min_n_buffer_for`` legitimately returns 0 (all-persistent
    # layout — every chunk resident on GPU, no offload/gather routes
    # through the pool), skip pool construction entirely. Allocating a
    # dormant 1-slot pool would burn S_chunk bytes of pinned host AND
    # S_chunk bytes of GPU memory outside the searched budget, which
    # the cost model and CPU/GPU gates are supposed to prevent (on
    # large models S_chunk can be 128 MiB+). The runtime's persistent
    # path never touches ``self.buffer_pool`` so leaving it as ``None``
    # is correctness-safe; ChunkManager's pool-touching methods all
    # early-return for persistent chunks.
    pinned_host: "PinnedHostMemory | None"
    buffer_pool: "BufferPool | None"
    if n_buffer == 0:
        pinned_host = None
        buffer_pool = None
    else:
        pinned_host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=layout.S_chunk)
        buffer_pool = BufferPool(
            n_buffer=n_buffer,
            S_chunk=layout.S_chunk,
            pinned_host=pinned_host,
            device=device,
        )

    # Compute the effective persistent set FIRST so the param
    # partitioning + the ChunkManager construction agree on which
    # chunks are persistent.
    #
    # The runtime resident set is ``{0..n_persist-1} ∪
    # layout.mandatory_persistent``. ``layout.mandatory_persistent`` is
    # populated once by :func:`build_layout` and records every chunk
    # containing at least one non-block param (e.g. ``model.norm.weight``,
    # an untied ``lm_head``); the block-granularity scheduler cannot
    # gather such chunks on its own, so they MUST stay GPU-resident.
    # Any non-block chunk at ``cid >= n_persist`` MUST therefore land
    # in the GPU optimizer's param list, not per-chunk CPU FusedAdam:
    # ``materialize_offload`` only offloads chunks in
    # ``_non_persistent_ids``, so a high-cid non-block chunk (e.g. an
    # untied lm_head at the tail of N_chunk) would otherwise be routed
    # to CPU adam against GPU-resident params.
    #
    # ``cfg.n_persist`` here is unchanged from the search's pick — it
    # remains the *prefix length* the search chose, not a count of the
    # full augmented set. ``ChunkManager.mark_persistent(n_persist)``
    # honours ``layout.mandatory_persistent`` natively, so no in-place
    # mutation of ``chunk_manager._persistent_ids`` is needed below.
    effective_persistent_ids: frozenset[ChunkId] = layout.effective_persistent_ids(
        n_persist
    )
    if layout.mandatory_persistent:
        LOG.info(
            "ProTrain: %d chunks %s pinned by layout.mandatory_persistent "
            "(non-block params the block-granularity scheduler cannot "
            "gather on its own); residency = prefix[0..%d) ∪ mandatory",
            len(layout.mandatory_persistent),
            sorted(layout.mandatory_persistent),
            n_persist,
        )

    # Partition params: persistent chunks get the GPU optimizer, the rest
    # get per-chunk CPU FusedAdam adapters keyed on ChunkId.
    params_by_name: dict[str, nn.Parameter] = dict(model.named_parameters())
    persistent_params: list[nn.Parameter] = []
    cpu_params_per_chunk: dict = {}

    for cid, chunk_param_ids in enumerate(layout.chunks):
        chunk_params = [
            params_by_name[str(pid)]
            for pid in chunk_param_ids
            if str(pid) in params_by_name
        ]
        if cid in effective_persistent_ids:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[cid] = chunk_params

    # Adam hyperparameters are owned by the optimizer wrapper; seed with
    # harmless defaults here. ``protrain_optimizer_wrapper`` will rebuild
    # these adapters with the user's real LR/betas, so this instance is
    # transient — we still allocate it so the chunk manager has a live
    # reference during the smoke-test smoke path.
    #
    # BUG 3 FIX: ``CpuFusedAdamAdapter`` construction is deferred to
    # AFTER ``chunk_manager.materialize_offload()`` below. Before
    # offload, the non-persistent chunk params are full-size GPU
    # tensors; after offload they are zero-element GPU placeholders
    # whose *real* weights live in ``chunk_manager._cpu_slots``. The
    # lazy CPU-Adam state init (``torch.zeros_like(p.data, device='cpu')``)
    # runs on the first ``step`` call — by which point
    # ``_ensure_cpu_grads_attached`` has repointed ``p.data`` at the CPU
    # shard — so what matters is that the adapter's ``param_groups``
    # reference the right ``nn.Parameter`` objects, not what ``p.data``
    # currently points at. The previous ordering (adapter built
    # pre-offload) was benign in the p.data sense but risked a CUDA
    # initialization hazard if DeepSpeed ever cached pointers on the
    # GPU tensor; deferring is the safe invariant.
    gpu_optim: GpuFusedAdamAdapter | None = None
    if persistent_params:
        gpu_optim = GpuFusedAdamAdapter(params=persistent_params, lr=1e-4)

    # ---- Distributed context + M7 zero3_shard decision -----------------
    # Auto-detect world_size / rank from the active process group;
    # default to single-rank when no group is up. ``zero3_shard`` was
    # already resolved above the search call so it could flow through
    # ``HardwareProfile.zero3_shard`` into the cost model; re-use that
    # decision here for the ChunkManager constructor. The ChunkManager
    # silently degrades zero3_shard to False when world_size == 1, so
    # the auto-detect path is safe on single-rank hosts too.
    _ws = 1
    _rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _ws = int(torch.distributed.get_world_size())
        _rank = int(torch.distributed.get_rank())
    _zero3 = bool(hardware_profile.zero3_shard) and (_ws > 1)
    LOG.info(
        "ProTrain: distributed context world_size=%d rank=%d zero3_shard=%s "
        "(requested=%s)",
        _ws,
        _rank,
        _zero3,
        zero3_shard,
    )

    chunk_manager = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=buffer_pool,
        cpu_optim=None,  # wired in after materialize_offload (BUG 3)
        gpu_optim=gpu_optim,
        device=device,
        world_size=_ws,
        rank=_rank,
        zero3_shard=_zero3,
    )

    # The non-block-chunk pinning that earlier versions performed here
    # in-place on ``chunk_manager._{,non_}persistent_ids`` is now built
    # into ``ChunkManager.__init__`` via
    # ``layout.effective_persistent_ids(n_persist)``. Reasoning for the
    # pin (preserved here for the comment trail):
    #
    #   a) The block-granularity scheduler only knows about chunks
    #      listed in ``layout.block_to_chunks``. Pure non-block chunks
    #      (the trivial case — all their params are non-block) are
    #      never gathered by any hook; if offloaded they'd be
    #      zero-sized during forward.
    #   b) Mixed chunks (e.g. the last block's chunk that was greedy-
    #      filled with the final ``model.norm.weight``) ARE gathered by
    #      the block-post hook, but the block-post hook ALSO releases
    #      them since they're not in the next block's chunk set —
    #      which leaves the non-block param empty by the time
    #      ``LlamaModel.forward`` calls ``self.norm(...)`` after the
    #      last block's forward-post hook fires.
    #
    # The fix in both cases is the same: keep chunks with any non-block
    # param GPU-resident. Cost is bounded by ``S_chunk`` per such chunk;
    # for Llama it's typically 2 chunks ≈ 256 MB. Tracked via
    # ``layout.mandatory_persistent``; surfaces to the search and cost
    # model through ``ChunkLayout.effective_persistent_ids``.

    # Sanity check: the chunk manager's runtime residency must match
    # the partitioning we used to build ``persistent_params`` /
    # ``cpu_params_per_chunk`` above. Drift here would silently misroute
    # a chunk between GPU and CPU optimisers.
    assert chunk_manager._persistent_ids == set(effective_persistent_ids), (
        "ChunkManager residency drift: expected "
        f"{sorted(effective_persistent_ids)}, got "
        f"{sorted(chunk_manager._persistent_ids)}"
    )

    # ---- peak-prediction calibration ------------------------------------
    # The cost/memory.py estimator approximates persistent model state as
    # ``n_persist * S_chunk`` — a tight upper bound when chunks pack
    # snugly to S_chunk, but a loose one when the layout leaves many
    # chunks partially filled (common for Llama-7B: avg chunk density
    # ~80% of S_chunk). For the integration-test peak-tolerance check
    # to land within the paper's stated "up to 10% overestimate" window
    # we recompute the model-state-present term using the *actual*
    # per-chunk byte footprint, then preserve the estimator's F_bm
    # (fragmentation + activation + inter/intra-op delta) component.
    calibrated_peak = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=result.predicted_peak_bytes,
        layout=layout,
        chunk_manager=chunk_manager,
        cfg=result.cfg,
        trace=trace,
        block_map=result.block_map,
    )
    if calibrated_peak != result.predicted_peak_bytes:
        LOG.info(
            "ProTrain: peak prediction calibrated %.2f -> %.2f GB "
            "using actual per-chunk byte footprint",
            result.predicted_peak_bytes / (1 << 30),
            calibrated_peak / (1 << 30),
        )
        # ``cfg.n_persist`` continues to mean "prefix length the search
        # chose". Earlier versions of this site collapsed it into
        # ``len(chunk_manager._persistent_ids)`` — the augmented set
        # including ``layout.mandatory_persistent`` — which made the
        # value disagree with how every other consumer
        # (``min_n_buffer_for``, ``model_state_present_bytes``,
        # ``block_map_runtime_admissible``, telemetry) reads it. Now
        # the augmented set is plumbed through ``layout.mandatory_persistent``;
        # the prefix is preserved here verbatim.
        result = SearchResult(
            cfg=CostConfig(
                n_persist=result.cfg.n_persist,
                n_buffer=result.cfg.n_buffer,
                n_swap=result.cfg.n_swap,
                n_checkpoint=result.cfg.n_checkpoint,
                # Option B: preserve the n_offload axis through peak
                # calibration. Pre-Option-B this rebuild silently
                # dropped n_offload because the field didn't exist;
                # without this carry-over an explicit
                # n_offload_override would be erased the moment a
                # block_map calibration fired (M5 follow-up, see
                # BLOCK_MODE_OFFLOAD_DESIGN.md §M5).
                n_offload=result.cfg.n_offload,
            ),
            block_map=result.block_map,
            predicted_peak_bytes=calibrated_peak,
            predicted_iter_s=result.predicted_iter_s,
        )

    # ---- 4.5: materialize the init-time chunk offload (M4.5 Gap 1) -----
    # Physically move every non-persistent chunk's param data to pinned
    # CPU memory and install the per-param grad hooks (Gap 2). This must
    # happen BEFORE step 5 (block wrap) / step 6 (hook install) so the
    # first forward sees the correct GPU residency picture and the grad
    # hooks are live by the time autograd starts accumulating.
    alloc_before = (
        torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    )
    freed = chunk_manager.materialize_offload()
    alloc_after = (
        torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    )
    LOG.info(
        "ProTrain: materialize_offload freed %.2f GB (reported), "
        "alloc %.2f -> %.2f GB (torch measured)",
        freed / (1 << 30),
        alloc_before / (1 << 30),
        alloc_after / (1 << 30),
    )
    _sys2.stderr.write(
        f"[protrain] materialize_offload: freed {freed / 1e9:.2f}GB "
        f"(alloc {alloc_before / 1e9:.2f}->{alloc_after / 1e9:.2f}GB)\n"
    )
    _sys2.stderr.flush()

    # ---- 4.6: build the CPU FusedAdam adapter (post-offload) ------------
    # BUG 3 FIX: now that ``materialize_offload`` has allocated the pinned
    # CPU shards and installed per-param grad hooks, build the CPU Adam
    # adapter with references to the same ``nn.Parameter`` objects the
    # hooks will repoint to CPU storage before calling step. The adapter
    # is "transient" (``protrain_optimizer_wrapper`` rebuilds it at the
    # user's real hyperparams) but we still need one live here so the
    # chunk manager has something to drive during smoke tests.
    # M7: for sharded non-persistent chunks, the CPU Adam updates each
    # region's flat shard_param (one per :class:`_DtypeRegion`) rather
    # than the user-facing param list. Homogeneous-dtype chunks have
    # one region and behave exactly like the pre-followup single-param
    # case; mixed-dtype chunks expose one shard_param per region.
    cpu_params_per_chunk_for_optim: dict = {}
    for cid, chunk_params in cpu_params_per_chunk.items():
        shard_state = chunk_manager._chunk_shards.get(cid)  # type: ignore[attr-defined]
        if shard_state is not None and shard_state.regions:
            cpu_params_per_chunk_for_optim[cid] = [
                r.shard_param for r in shard_state.regions
            ]
        else:
            cpu_params_per_chunk_for_optim[cid] = chunk_params

    cpu_optim: CpuFusedAdamAdapter | None = None
    if any(params for params in cpu_params_per_chunk_for_optim.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk_for_optim,
                lr=1e-4,
            )
        except (ImportError, Exception) as err:  # noqa: BLE001 - see below
            # CpuFusedAdamAdapter can fail with more than ``ImportError``:
            # DeepSpeed raises ``CUDAMismatchException`` (not an
            # ``ImportError`` subclass) when the system nvcc and torch's
            # cu-version disagree. We degrade gracefully in both cases —
            # persistent chunks still run fused GPU Adam, non-persistent
            # chunks fall through to the in-line torch.optim path inside
            # the optimizer wrapper. The warning surfaces the root cause
            # so users know they're not getting the async overlap.
            #
            # IMPORTANT: render ``err`` to a string before logging — passing
            # the live exception object propagates ``err.__traceback__`` →
            # frame locals (which include large GPU param lists in this
            # scope) into the LogRecord. pytest log-capture retains those
            # records, leaking one full model footprint per failed attempt.
            err_repr = f"{type(err).__name__}: {err}"
            LOG.warning(
                "ProTrain: CPU FusedAdam unavailable (%s); non-persistent chunks "
                "will not get async CPU Adam. Install DeepSpeed with a matching "
                "CUDA toolkit (or set DS_SKIP_CUDA_CHECK=1) for full coverage.",
                err_repr,
            )
            del err
            cpu_optim = None
    chunk_manager.cpu_optim = cpu_optim

    eff_h2d, eff_d2h = effective_bw(result.cfg, hardware_profile)

    scheduler = Scheduler(
        chunk_manager=chunk_manager,
        block_map=result.block_map,
        layout=layout,
        effective_h2d_bps=eff_h2d,
        effective_d2h_bps=eff_d2h,
    )

    # ---- 5. wrap blocks -------------------------------------------------
    # Locate the parent ModuleList(s) so we can swap in the wrapped blocks
    # in-place. Encoder-decoder models have two ModuleLists (encoder.block
    # and decoder.block); ``_find_block_parent_map`` returns one per block.
    block_parent_map = _find_block_parent_map(model, blocks)
    for idx, block in enumerate(blocks):
        mode = result.block_map.get(BlockId(idx))
        if mode is None:
            continue
        wrapped_block = wrap_block(block, mode)
        if wrapped_block is not block:
            parent = block_parent_map.get(id(block))
            if parent is not None:
                # Find the slot index within the parent ModuleList
                # (cannot reuse ``idx`` — that's the global block index,
                # which differs from the within-tree position for
                # decoder blocks of an encoder-decoder model).
                for slot, child in enumerate(parent):
                    if child is block:
                        parent[slot] = wrapped_block
                        break
            blocks[idx] = wrapped_block

    # ---- 5.5. wire up the activation SWAP pool --------------------------
    # When the searcher (or an explicit override) selects ``n_swap > 0``,
    # build a single :class:`ActivationSwapPool` sized to hold
    # ``n_swap * prefetch_depth`` activation slots in pinned host memory,
    # then attach the pool + scheduler's ``_swap_stream`` to every
    # :class:`SwappedBlock`. The wrapper degrades to identity-pass
    # autograd if the pool is None — useful for CPU-only test paths,
    # but a configuration error in production.
    if result.cfg.n_swap > 0:
        from axolotl.integrations.protrain.types import BlockMode as _BM_swap

        # Worst-case activation bytes across the swap-band. Reading from
        # ``trace.activation_sizes`` (per-block) keeps this aligned with
        # the cost model's ``estimate_cpu_footprint`` accounting.
        max_act_bytes = 0
        for bid, mode in result.block_map.items():
            if mode is _BM_swap.SWAP:
                act = trace.activation_sizes.get(bid, 0)
                if act > max_act_bytes:
                    max_act_bytes = int(act)
        if max_act_bytes <= 0:
            LOG.warning(
                "ProTrain: result.cfg.n_swap=%d but no SWAP block has "
                "non-zero activation_sizes; skipping swap-pool construction",
                result.cfg.n_swap,
            )
        else:
            from axolotl.integrations.protrain.block.swap_pool import (
                DEFAULT_SLOTS_PER_BLOCK,
                ActivationSwapPool,
            )
            from axolotl.integrations.protrain.cost.memory import (
                SWAP_PREFETCH_DEPTH,
            )

            # Each slot must be large enough for the worst-case single
            # saved tensor inside any SWAP block. The trace records only
            # the per-block AGGREGATE (sum across all saved tensors) —
            # there is no per-tensor breakdown. The previous formula
            # ``ceil(aggregate / slots_per_block)`` modelled a uniform
            # split, but real transformer blocks have skewed tensor
            # distributions (the residual stream alone can dominate
            # ~1/3-1/2 of the aggregate while small Q/K projections
            # share the remainder). When SWAP encounters a saved tensor
            # larger than the AVERAGE-derived slot, ``slot_view.view(
            # dtype).copy_(tensor)`` raises ``RuntimeError`` at runtime.
            # Until per-tensor profiling lands, size every slot to the
            # full per-block aggregate. The pool is over-provisioned
            # (worst case ~K× larger than necessary) but cannot fail at
            # runtime regardless of the saved-tensor size distribution.
            # The cost model in ``cost/memory.estimate_cpu_footprint``
            # uses the same formula so the searcher's CPU gate stays
            # aligned with the actual runtime allocation.
            slots_per_block = DEFAULT_SLOTS_PER_BLOCK
            # Floor at 1 byte to satisfy the pool's positive-size invariant.
            per_slot = max(1, int(max_act_bytes))
            swap_pool = ActivationSwapPool(
                n_swap=result.cfg.n_swap,
                slot_bytes=per_slot,
                prefetch_depth=SWAP_PREFETCH_DEPTH,
                slots_per_block=slots_per_block,
            )
            scheduler.swap_pool = swap_pool
            for block in blocks:
                if getattr(block, "_protrain_wrapped_mode", None) is _BM_swap.SWAP:
                    block.attach_runtime(swap_pool, scheduler.swap_stream)
            LOG.info(
                "ProTrain: SWAP pool wired — %d slots × %d bytes = %.2f MB pinned",
                swap_pool.n_slot,
                swap_pool.slot_bytes,
                swap_pool.total_bytes / (1 << 20),
            )

    # ---- 6. install hooks ----------------------------------------------
    handles = install_hooks(
        model=model,
        chunk_manager=chunk_manager,
        block_map=result.block_map,
        scheduler=scheduler,
    )

    # ``capacity_bytes`` is unused inside the helper — kept in the
    # signature for symmetry with the wrapper's call site so a future
    # extension that derates by capacity (e.g. peak vs. budget headroom)
    # can read it without refactoring callers.
    del capacity_bytes  # silence linter

    return chunk_manager, scheduler, list(handles), result


def protrain_model_wrapper(
    model: nn.Module,
    model_config: object,  # noqa: ARG001 — accepted for API symmetry with the plan
    hardware_profile: HardwareProfile,
    *,
    batch_size: int,
    seq_len: int,
    capacity_bytes: int | None = None,
    cpu_capacity_bytes: int | None = None,
    cache_dir: str | None = None,
    force_all_persistent: bool = False,
    n_persist_override: int | None = None,
    n_buffer_override: int | None = None,
    n_swap_override: int | None = None,
    n_checkpoint_override: int | None = None,
    n_offload_override: int | None = None,
    zero3_shard: bool | None = None,
    auto_mode: bool = False,
    target_device: "torch.device | str | int | None" = None,
) -> WrappedModel:
    """Compose the ProTrain runtime around a standard ``nn.Module``.

    Parameters
    ----------
    model:
        Any standard ``nn.Module``. Must be on GPU by the time this is
        called; the profiler and all buffers are allocated on the same
        device as ``next(model.parameters()).device``.
    model_config:
        Reserved. The plugin path (M5) will use this to pick up
        ZeRO-related options; the M4b wrapper does not consult it.
    hardware_profile:
        Static hardware descriptor — see
        :class:`~axolotl.integrations.protrain.types.HardwareProfile`.
    batch_size / seq_len:
        Used for both the profiler invocation and the cache key.
    capacity_bytes:
        Override the GPU memory budget the searcher should respect.
        When ``None``, defaults to
        ``hardware_profile.gpu_memory_bytes - 2 GiB`` to leave headroom
        for the CUDA context + PyTorch allocator.
    cpu_capacity_bytes:
        Per-rank pinned CPU RAM budget the searcher should treat as a
        HARD feasibility filter. Configs whose
        :func:`~axolotl.integrations.protrain.cost.memory.estimate_cpu_footprint`
        exceeds this value are dropped before runtime evaluation, so
        the picked config is guaranteed to fit BOTH the GPU and CPU
        envelopes. When ``None`` (default), the wrapper auto-derives
        ``psutil.virtual_memory().available // hw.gpu_count - 2 GiB``;
        if psutil is not installed, the filter is disabled and a
        warning is logged. Pass an explicit ``int`` to override the
        auto-derivation, or pass an explicit ``int(<huge>)`` (or a
        negative dummy value via the wrapping plugin) to deactivate
        when the auto value over-restricts on machines with NUMA-aware
        allocators. Complements the :func:`_select_mode` auto-mode
        layer: the SEARCH filter gates which configs are even
        evaluable; auto-mode then picks between feasible cfgs that
        already passed both gates.
    cache_dir:
        Override the profiler cache root. When provided (non-None) the
        profiler stores / loads traces under
        ``<cache_dir>/protrain/profiler``, taking precedence over the
        ``XDG_CACHE_HOME`` env var. When ``None`` (default), resolution
        falls back to ``profiler.cache._cache_root`` (XDG-style).
    force_all_persistent:
        When True, skip the exhaustive searcher and synthesize a
        ``SearchResult`` that forces every chunk to stay GPU-resident
        (``n_persist = N_chunk``, ``n_swap = 0``,
        ``n_checkpoint = N_block``). This is the M5 recommended mode
        for LoRA on a single 24 GB card until the M4.5 runtime
        primitives (init-time chunk offload, per-param grad offload)
        land — search-picked configs that expect CPU-hosted chunks
        currently OOM because the physical offload is not yet wired.
    n_persist_override / n_buffer_override / n_swap_override / n_checkpoint_override:
        Debug escape hatches. When *all four* are set, the searcher is
        skipped and a synthetic ``SearchResult`` is built from the
        explicit values. A single override in isolation is ignored (the
        searcher's picks stay consistent across the 4-tuple); this is
        documented on the pydantic fields.
    n_offload_override:
        Optional Option B knob (see ``BLOCK_MODE_OFFLOAD_DESIGN.md``)
        plumbed alongside the 4-tuple override path. When omitted (or
        ``None``) defaults to 0 — pre-Option-B callers see identical
        behaviour. When the four-tuple override path is active and
        ``n_offload_override`` is non-zero, that many block positions
        are tagged ``BlockMode.OFFLOAD`` by ``assign_modes`` (placed in
        the unopt-late tail before NONE — see ``layout_rules.py``).
        Use this to drive a "no-recompute on non-persistent blocks"
        config: set ``n_checkpoint_override=0`` and
        ``n_offload_override = N_block - n_swap_override``. Bounds:
        ``0 <= n_offload <= N_block - n_swap - n_checkpoint``; outside
        this range the override path raises ``ValueError`` to mirror
        the searcher's enumeration.
    zero3_shard:
        M7 ZeRO-3 activation. When ``None`` (default) the wrapper
        auto-detects: shard iff
        ``torch.distributed.get_world_size() > 1`` AND
        ``force_all_persistent`` is False. When explicitly True or
        False the caller override wins. Sharded mode requires a live
        ``torch.distributed`` process group AND the model must not be
        wrapped in DDP at training time (sharding is the grad-sync
        point itself; DDP would double-reduce).
    auto_mode:
        When True, the wrapper runs the searcher first and then calls
        :func:`_select_mode` to resolve ``(force_all_persistent,
        zero3_shard)`` from workload fit + per-rank CPU RAM. The
        caller's ``force_all_persistent`` / ``zero3_shard`` arguments
        are IGNORED on this path (they become explicit overrides only
        when ``auto_mode=False``). Designed to save users from the
        ZeRO-3 footgun surfaced by the M7 benchmark (0.70x throughput
        vs. 3.64x DDP on PCIe Gen3 4x 3090 when the model fits on GPU).
        Default is False on this direct entry point; the plugin sets it
        to True via ``ProTrainArgs.protrain_auto_mode``.
    target_device:
        Explicit per-rank target device for downstream GPU allocations
        (BufferPool, ChunkManager, profiler). When provided, all
        GPU-side state is allocated on this device and the model is
        moved here before profiling if it isn't already. Takes
        precedence over the ``model._protrain_target_device``
        metadata hint and over ``next(model.parameters()).device``.
        Pass ``None`` (default) to fall back to the hint or to the
        model's existing placement — the latter is the legacy contract
        for callers that hand in an already-placed model. The plugin
        path computes this from ``LOCAL_RANK`` so each rank's chunks
        land on its own GPU; ``None`` is also passed when
        ``hf_device_map`` is set so the wrapper inherits the
        device-mapped placement instead of collapsing it onto a
        single device.

    Returns
    -------
    WrappedModel
        Handle carrying the search result, chunk manager, scheduler,
        and the installed hook handles. The underlying ``model`` is
        returned in-place — no module swap.
    """
    import torch

    # ---- Device resolution -----------------------------------------------
    # Precedence:
    #   1. Explicit ``target_device`` kwarg (the plugin's preferred path —
    #      computed from LOCAL_RANK before any model placement happens).
    #   2. ``model._protrain_target_device`` metadata hint (lightweight
    #      handoff for callers that bypass the kwarg path).
    #   3. ``next(model.parameters()).device`` — back-compat fallback for
    #      callers that hand in an already-placed model and don't supply
    #      a hint.
    #   4. ``cuda:0`` if CUDA is available, else ``cpu``.
    #
    # The plugin no longer calls ``model.to()`` in ``post_model_load`` —
    # that was a footgun that either (a) eagerly materialized the full
    # model on a single device (defeating the chunk-offload promise) or
    # (b) silently swallowed an OOM and left the wrapper running on a
    # CPU model whose downstream allocations seeded the wrong device.
    # The wrapper now owns placement: when the model is on CPU but the
    # caller-supplied target is CUDA, we move the model here, with a
    # try/except that surfaces an actionable error rather than returning
    # a half-wired WrappedModel. The on-demand profiler path
    # (``OnDemandTensorMgr``) handles the "model exceeds device memory"
    # case downstream — see ``run_trace``'s ``engage_on_demand`` gate.
    resolved_target: torch.device | None = None
    if target_device is not None:
        resolved_target = torch.device(target_device)
    else:
        attr_target = getattr(model, "_protrain_target_device", None)
        if attr_target is not None:
            resolved_target = torch.device(attr_target)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = None

    if resolved_target is not None:
        device = resolved_target
    elif model_device is not None:
        device = model_device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If the caller asked for a CUDA target but the model is on a
    # different device (typically CPU under ``accelerate launch``, where
    # ``post_model_load`` fires before ``Accelerator.prepare()``), move
    # it now. ``hf_device_map``-loads pass ``target_device=None`` from
    # the plugin, so this branch only fires for the in-process and
    # accelerate-launch single-rank paths the plugin is actually
    # responsible for placing. We do NOT silently catch OOM here — a
    # failure on this move has no safe automatic recovery (downstream
    # GPU-side allocations would be wrong-sized for the actual device
    # state); the trainer-level catch handles it as an unrecoverable
    # config error with full context.
    if (
        device.type == "cuda"
        and model_device is not None
        and model_device != device
        and getattr(model, "hf_device_map", None) is None
    ):
        LOG.info(
            "ProTrain: moving model from %s to %s before profiling "
            "(target_device hint).",
            model_device,
            device,
        )
        model.to(device)

    # Gradient checkpointing + HF KV cache leads to recompute-time shape
    # mismatches (cache grows across calls; the recompute call sees a
    # different past_key_values length). Force use_cache=False if the model
    # exposes it — this is standard practice for training regardless of
    # ProTrain, and the CKPT block wrapper depends on it.
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is not None and getattr(cfg_obj, "use_cache", False):
        LOG.info(
            "ProTrain: forcing model.config.use_cache=False for CKPT compatibility"
        )
        cfg_obj.use_cache = False

    # ---- 1. profile (cached) --------------------------------------------
    cache_key = ProfilerCacheKey(
        arch_hash=_arch_hash(model),
        bs=batch_size,
        seq=seq_len,
        sku=_sku(device),
        world=hardware_profile.gpu_count,
    )
    trace = load_cached_trace(cache_key, cache_dir=cache_dir)
    if trace is None:
        import sys as _sys

        LOG.info(
            "ProTrain profiler cache miss for %s — running trace (bs=%d seq=%d)",
            cache_key.fingerprint()[:12],
            batch_size,
            seq_len,
        )
        _sys.stderr.write(
            "[protrain] profiler cache miss — running backward-aware trace\n"
        )
        _sys.stderr.flush()
        # Backward-aware profile (paper §3.2 / App A.2 fidelity, since
        # TRACE_VERSION 19). The paper's profiler captures both forward
        # and backward windows because peak memory typically occurs in
        # backward (retained activations + grad buffers + CKPT recompute
        # all overlap). Skipping backward forced ``cost/runtime.py`` to
        # derive ``t_bwd`` analytically from ``t_fwd`` and the cost
        # model's bwd peak to fall back to the trainable-fraction
        # heuristic — both weakened searcher ranking on borderline
        # configs.
        #
        # OOM guard: the original concern (running ``loss.backward()``
        # on a 7B-class model blew the 24 GiB card before chunk offload
        # could engage) is now mitigated by ``OnDemandTensorMgr``'s
        # post-630e5dd4 spill semantics — GPU-resident params actually
        # release their storage during the on-demand context, and
        # ``saved_tensors_hooks`` spills retained activations to CPU
        # for the duration of forward+backward. ``run_trace`` engages
        # on-demand whenever the model state (params + grads + Adam)
        # exceeds 60% of device memory; on smaller models the unwrapped
        # backward is well within budget. The hot-loop steady backward
        # is additionally guarded with a per-iter try/except that
        # falls back to the analytical bwd_fwd ratio if any iter raises
        # — no regression vs. the pre-v19 path on the OOM edge case.
        profiler_cfg = ProfilerConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            device=str(device),
            include_backward=True,
            on_demand=True,
            world_size=int(hardware_profile.gpu_count),
        )
        batch = _dummy_batch(model, batch_size, seq_len, device)
        trace = run_trace(model, batch, profiler_cfg)
        _sys.stderr.write(
            f"[protrain] trace done: {len(trace.op_order)} ops, "
            f"{len(trace.activation_sizes)} blocks\n"
        )
        _sys.stderr.flush()
        save_cached_trace(cache_key, trace, cache_dir=cache_dir)
    else:
        LOG.info("ProTrain profiler cache hit for %s", cache_key.fingerprint()[:12])

    # ---- 2. layout ------------------------------------------------------
    import sys as _sys2

    _sys2.stderr.write("[protrain] building layout\n")
    _sys2.stderr.flush()
    blocks, block_spans = _build_block_spans(model)
    exec_order = _param_exec_order(model, block_spans, trace)

    # Derive S_chunk from a {ParamId -> bytes} map. Pass exec_order +
    # block_spans so the grid-search simulation honors the same block-sealing
    # / contiguity rules ``build_layout`` will use — App B.1 says ProTrain
    # "simulates memory waste across various chunk sizes" and selects the
    # one minimizing it; with the block info wired through, the simulation
    # matches the actual layout placement bit-for-bit.
    param_bytes: dict[ParamId, int] = {
        cast(ParamId, name): int(p.numel()) * int(p.element_size())
        for name, p in model.named_parameters()
    }
    s_chunk = pick_S_chunk(
        param_bytes,
        exec_order=exec_order,
        block_spans=block_spans,
    )

    layout = build_layout(
        model=model,
        exec_order=exec_order,
        S_chunk=s_chunk,
        block_spans=block_spans,
    )
    _sys2.stderr.write(
        f"[protrain] layout built: S_chunk={layout.S_chunk} N_chunk={layout.N_chunk}\n"
    )
    _sys2.stderr.flush()

    # ---- 3. search (or synthesize) -------------------------------------
    if capacity_bytes is None:
        capacity_bytes = max(
            0, int(hardware_profile.gpu_memory_bytes) - _DEFAULT_HEADROOM_BYTES
        )

    # Auto-derive the search-time CPU feasibility budget when the caller
    # did not provide one. This is a HARD search filter (configs whose
    # estimated per-rank pinned CPU footprint exceeds this value are
    # dropped before runtime evaluation), distinct from and complementary
    # to the auto-mode selector below — see ``_select_mode``.
    # ``_default_cpu_capacity_for_search`` returns ``None`` when psutil
    # isn't installed (logs a warning) so the searcher falls back to its
    # GPU-only behaviour.
    if cpu_capacity_bytes is None:
        cpu_capacity_bytes = _default_cpu_capacity_for_search(
            hardware_profile.gpu_count
        )

    # Early world-size probe — the mode selector + zero3_shard plumbing
    # both need this before the search runs.
    _ws_early = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _ws_early = int(torch.distributed.get_world_size())

    # Stash the caller's raw intent before the auto-selector potentially
    # rewrites the effective flags. The selector is applied AFTER
    # search() returns; the search itself runs against a hardware
    # profile whose ``zero3_shard`` flag is resolved a few lines below
    # to keep the CPU-capacity hard gate from preempting the auto-mode
    # selector — see the block immediately following the auto-mode
    # short-circuit for the full rationale.
    _user_force_all_persistent = bool(force_all_persistent)
    _user_zero3_shard = zero3_shard

    if auto_mode:
        # On the auto path, disable the force_all_persistent short-circuit
        # below and let the searcher pick n_persist. If the fit is tight
        # the selector flips the mode post-search; if the fit is loose
        # the searcher lands at n_persist=N_chunk naturally, which is
        # already Mode A semantically (no runtime difference vs. the
        # force_all_persistent synthetic path). We also suppress an
        # explicit user ``zero3_shard=True`` for the hw profile here;
        # it gets re-evaluated after search + selector.
        if _user_force_all_persistent:
            LOG.info(
                "ProTrain auto-mode: user set force_all_persistent=True "
                "but auto-mode overrides explicit flags. Running searcher "
                "— will pick Mode A naturally if the workload fits on "
                "GPU. Set ``protrain_auto_mode: false`` to force-honour "
                "force_all_persistent=True."
            )
        force_all_persistent = False
        zero3_shard = False

    # Resolve the ZeRO-3 sharding flag early so we can propagate it into
    # ``HardwareProfile`` before the cost-model search runs. The same
    # rules as the later in-place re-check (post-materialize_offload)
    # apply here — auto-enable when ``world_size > 1`` AND
    # ``force_all_persistent`` is False, honour explicit caller
    # overrides otherwise. The ChunkManager additionally degrades to
    # False on single-rank hosts (so setting this True on ws=1 is a
    # no-op); we mirror that here for HW profile consistency.
    #
    # On the auto-mode multi-rank path we deliberately overstate
    # ``zero3_shard=True`` for the SEARCH-TIME hardware profile so the
    # ``cpu_capacity_bytes`` hard gate inside ``search()`` uses the
    # SHARDED (most-permissive) per-rank footprint. Otherwise the gate
    # would reject configs that fit under sharding before
    # ``_select_mode`` ever gets to enable Mode C. The post-search
    # selector (``_select_mode``) then re-evaluates both replicated and
    # sharded footprints against the actual per-rank RAM and either
    # picks the right mode or raises a clear RuntimeError; here we just
    # make sure the search itself doesn't preempt that decision. The
    # GPU peak filter is sharding-agnostic (see
    # ``cost/memory.estimate_peak``), so the searcher's pick of
    # ``n_persist`` is not distorted by this choice.
    if auto_mode and _ws_early > 1:
        _zero3_for_hw = True
    elif zero3_shard is None:
        _zero3_for_hw = (_ws_early > 1) and (not force_all_persistent)
    else:
        _zero3_for_hw = bool(zero3_shard) and (_ws_early > 1)
    # Propagate into the hardware_profile the searcher consumes. Replace
    # is cheap; HardwareProfile is frozen so we can't mutate in place.
    # We also plumb the trace's measured Adam throughputs into the
    # hardware_profile so ``cost/runtime.py`` consumes the empirical
    # rates rather than the hardcoded prior.
    from dataclasses import replace as _replace

    _hw_updates: dict = {}
    if _zero3_for_hw != hardware_profile.zero3_shard:
        _hw_updates["zero3_shard"] = _zero3_for_hw
    # Only overwrite Adam rates when the caller-provided profile doesn't
    # already carry them (i.e. tests that hand-craft a profile with a
    # specific rate keep their value). Non-zero trace measurement wins
    # over the default 0.0; 0.0 from the trace means the benchmark
    # couldn't run, and the runtime cost model will fall back.
    if (
        hardware_profile.cpu_adam_bytes_per_sec <= 0.0
        and trace.cpu_adam_bytes_per_sec > 0.0
    ):
        _hw_updates["cpu_adam_bytes_per_sec"] = trace.cpu_adam_bytes_per_sec
    if (
        hardware_profile.gpu_adam_bytes_per_sec <= 0.0
        and trace.gpu_adam_bytes_per_sec > 0.0
    ):
        _hw_updates["gpu_adam_bytes_per_sec"] = trace.gpu_adam_bytes_per_sec
    # Live SKU compute rate — measured fresh on the training device so the
    # cost model can scale per-op latencies when the trace was captured on
    # a different SKU (3090 vs 3090 Ti, etc.). Same-SKU runs see the same
    # value here as in trace.compute_rate_tflops, so the ratio is ~1.0.
    if hardware_profile.gpu_compute_tflops <= 0.0:
        try:
            _live_tflops = measure_compute_rate(int(getattr(device, "index", 0) or 0))
            if _live_tflops > 0.0:
                _hw_updates["gpu_compute_tflops"] = _live_tflops
        except Exception as _e:  # noqa: BLE001 - defensive
            LOG.debug(
                "measure_compute_rate live failed (%s); skipping SKU calibration", _e
            )
    # PCIe rates: overwrite the caller's hardcoded prior (usually 13e9 =
    # Gen3) with the profiler's measured H2D/D2H. A 3090 on PCIe Gen4 x16
    # sits around 50-56 GB/s — 4× the conservative default — and the
    # cost model's per-chunk comm is S_chunk / eff_h2d, so this flow-
    # through directly corrects the 7B over-prediction.
    if (
        hardware_profile.pcie_h2d_bps <= 13e9 + 1e6  # within 1MB of default
        and trace.pcie_h2d_bps > 13e9 + 1e6
    ):
        _hw_updates["pcie_h2d_bps"] = trace.pcie_h2d_bps
    if hardware_profile.pcie_d2h_bps <= 13e9 + 1e6 and trace.pcie_d2h_bps > 13e9 + 1e6:
        _hw_updates["pcie_d2h_bps"] = trace.pcie_d2h_bps
    if _hw_updates:
        hardware_profile = _replace(hardware_profile, **_hw_updates)

    # Snapshot the SEARCH-time hardware profile. The auto-mode path
    # below may re-stamp ``hardware_profile.zero3_shard`` after
    # ``_select_mode`` returns to reflect the RUNTIME mode, but the
    # phase-2 re-search must keep using the permissive (search-time)
    # profile to avoid filtering Mode-C-only candidates whose CPU
    # footprint only fits under sharding. On the non-auto-mode path
    # this snapshot is identical to ``hardware_profile`` end-to-end.
    search_hw_profile = hardware_profile

    n_block = max(1, len(trace.activation_sizes))

    all_overrides_set = all(
        v is not None
        for v in (
            n_persist_override,
            n_buffer_override,
            n_swap_override,
            n_checkpoint_override,
        )
    )

    if force_all_persistent:
        # Synthesize a SearchResult that pins every chunk on GPU and
        # uses activation checkpointing on every block. This is the M5
        # workaround for the two known M4.5 runtime gaps (init-time
        # chunk offload, per-param grad offload) — see DESIGN.md and
        # the M4 integration xfail. The cost model is skipped; predicted
        # numbers are filled with zeros so downstream consumers don't
        # misread them as real predictions.
        synth_cfg = CostConfig(
            n_persist=layout.N_chunk,
            n_buffer=min_n_buffer_for(layout, layout.N_chunk),
            n_swap=0,
            n_checkpoint=n_block,
        )
        block_map = assign_modes(n_swap=0, n_checkpoint=n_block, N_block=n_block)
        result = SearchResult(
            cfg=synth_cfg,
            block_map=block_map,
            predicted_peak_bytes=0,
            predicted_iter_s=0.0,
        )
        LOG.warning(
            "ProTrain: force_all_persistent=True — bypassing searcher. "
            "n_persist=%d n_buffer=%d n_swap=0 n_checkpoint=%d. "
            "All model state stays GPU-resident; activations rely on CKPT. "
            "This is the documented workaround for the M4.5 runtime gaps.",
            synth_cfg.n_persist,
            synth_cfg.n_buffer,
            synth_cfg.n_checkpoint,
        )
        _sys2.stderr.write(f"[protrain] force_all_persistent: cfg={result.cfg}\n")
        _sys2.stderr.flush()
    elif all_overrides_set:
        # Explicit 4-tuple override path — still skip the searcher but
        # honour the caller's exact knob selection. Bounds-check is
        # mandatory; the searcher normally enforces these.
        assert n_persist_override is not None
        assert n_buffer_override is not None
        assert n_swap_override is not None
        assert n_checkpoint_override is not None

        n_persist = int(n_persist_override)
        n_buffer = int(n_buffer_override)
        n_swap = int(n_swap_override)
        n_checkpoint = int(n_checkpoint_override)
        # Option B: plumb the optional ``n_offload`` knob through the
        # override path. Defaults to 0 to preserve pre-Option-B
        # behaviour for callers that omit the kwarg. See
        # ``BLOCK_MODE_OFFLOAD_DESIGN.md`` §3.6 / §7 (M5).
        n_offload = int(n_offload_override) if n_offload_override is not None else 0

        if not (0 <= n_persist <= layout.N_chunk):
            raise ValueError(
                f"n_persist_override={n_persist} out of range [0, {layout.N_chunk}]"
            )
        if n_buffer < 0:
            raise ValueError(f"n_buffer_override must be >= 0, got {n_buffer}")
        if not (0 <= n_swap <= n_block):
            raise ValueError(f"n_swap_override={n_swap} out of range [0, {n_block}]")
        if not (0 <= n_checkpoint <= n_block - n_swap):
            raise ValueError(
                f"n_checkpoint_override={n_checkpoint} incompatible "
                f"with n_swap_override={n_swap} (N_block={n_block})"
            )
        if not (0 <= n_offload <= n_block - n_swap - n_checkpoint):
            raise ValueError(
                f"n_offload_override={n_offload} incompatible with "
                f"n_swap_override={n_swap} + "
                f"n_checkpoint_override={n_checkpoint} (N_block={n_block}); "
                f"valid range is [0, {n_block - n_swap - n_checkpoint}]"
            )
        synth_cfg = CostConfig(
            n_persist=n_persist,
            n_buffer=n_buffer,
            n_swap=n_swap,
            n_checkpoint=n_checkpoint,
            n_offload=n_offload,
        )
        block_map = assign_modes(
            n_swap=n_swap,
            n_checkpoint=n_checkpoint,
            N_block=n_block,
            n_offload=n_offload,
        )

        # Replicate the searcher's two runtime-safety invariants. Without
        # these, the override path can ship configs that the searcher
        # would never select — e.g. an n_buffer too small for the
        # scheduler's lookahead prefetch (current-block ∪ next-block
        # non-persistent chunks must fit simultaneously) or a block_map
        # where a NONE block owns offloaded chunks (no activation-save
        # mechanism — autograd's saved tensors hold direct GPU storage
        # refs that the chunk pool's slot reuse will clobber). CKPT,
        # OFFLOAD and SWAP all tolerate non-persistent chunks (CKPT
        # recomputes; OFFLOAD re-gathers via saved-tensors-hook; SWAP
        # persists each saved tensor to a pinned-CPU pool slot decoupled
        # from param.data — see ``block_map_runtime_admissible`` and
        # the §6.6 SWAP × non-persistent lift in
        # ``BLOCK_MODE_OFFLOAD_DESIGN.md``).
        min_buffer = min_n_buffer_for(layout, n_persist)
        if n_buffer < min_buffer:
            raise ValueError(
                f"n_buffer_override={n_buffer} below scheduler minimum "
                f"{min_buffer} for n_persist={n_persist} on this layout "
                f"(N_chunk={layout.N_chunk}). The lookahead prefetch "
                "needs the union of current+next non-persistent chunks "
                "to fit in the pool simultaneously."
            )
        if not block_map_runtime_admissible(layout, block_map, n_persist):
            raise ValueError(
                f"override block_map for n_swap={n_swap} n_checkpoint={n_checkpoint} "
                f"n_offload={n_offload} is runtime-unsafe at n_persist={n_persist}: "
                "at least one block owns non-persistent chunks but is in NONE mode. "
                "NONE installs no activation-save hooks, so PyTorch's autograd "
                "saved-tensors hold direct GPU storage refs that the chunk pool's "
                "slot reuse will clobber. Use CKPT (recompute), OFFLOAD (saved-"
                "tensors-hook re-gather), or SWAP (saved tensors persisted to a "
                "pinned-CPU pool decoupled from param.data) for non-persistent "
                "blocks — or raise n_persist to make those blocks fully resident."
            )

        result = SearchResult(
            cfg=synth_cfg,
            block_map=block_map,
            predicted_peak_bytes=0,
            predicted_iter_s=0.0,
        )
        LOG.warning(
            "ProTrain: explicit knob override path — bypassing searcher. cfg=%s",
            synth_cfg,
        )
        _sys2.stderr.write(f"[protrain] explicit override: cfg={result.cfg}\n")
        _sys2.stderr.flush()
    else:
        _sys2.stderr.write(
            f"[protrain] running exhaustive search (N_chunk={layout.N_chunk}, "
            f"N_block={n_block})\n"
        )
        _sys2.stderr.flush()
        result = search(
            trace,
            layout,
            int(capacity_bytes),
            hardware_profile,
            cpu_capacity_bytes=cpu_capacity_bytes,
        )
        _sys2.stderr.write(
            f"[protrain] search done: cfg={result.cfg} "
            f"peak={result.predicted_peak_bytes / 1e9:.2f}GB "
            f"iter={result.predicted_iter_s:.3f}s\n"
        )
        _sys2.stderr.flush()

    # ---- 3.5: auto-mode selection (M7 follow-up) -----------------------
    # With the searcher's ``n_persist`` pick in hand, resolve the real
    # (force_all_persistent, zero3_shard) pair from workload fit +
    # per-rank CPU RAM. See ``_select_mode`` for the decision tree and
    # the DESIGN.md §Multi-GPU measured throughput ordering that
    # motivates the default (A > B > C on PCIe Gen3 3090).
    if auto_mode:
        cpu_ram = _cpu_ram_per_rank_bytes(_ws_early)
        if cpu_ram == 0 and _ws_early > 1:
            LOG.warning(
                "ProTrain auto-mode: could not probe CPU RAM via psutil or "
                "/proc/meminfo. Treating per-rank RAM as 0 bytes — the "
                "selector will prefer Mode A (force_all_persistent) and "
                "raise if the model needs offload. Set "
                "``protrain_auto_mode: false`` and pick the mode "
                "explicitly on exotic topologies."
            )
        auto_force_persistent, auto_zero3 = _select_mode(
            search_result=result,
            layout=layout,
            hw=hardware_profile,
            world_size=_ws_early,
            cpu_ram_per_rank_bytes=cpu_ram,
            auto_mode=True,
            user_force_all_persistent=_user_force_all_persistent,
            user_zero3_shard=_user_zero3_shard,
        )

        # Warn if the user set an explicit flag that the selector is
        # overriding. This is the key safety check for the M7 footgun:
        # users who requested ZeRO-3 on a workload that fits in Mode A
        # should learn they're leaving throughput on the table.
        if _user_zero3_shard is True and not auto_zero3 and _ws_early > 1:
            LOG.warning(
                "ProTrain auto-mode: user set zero3_shard=True but the "
                "workload fits in Mode A (force_all_persistent). "
                "Auto-mode picked Mode A for better throughput — on "
                "PCIe Gen3 RTX 3090, DDP+Mode_A gives ~3.6x scaling vs "
                "ZeRO-3's ~0.7x. Set ``protrain_auto_mode: false`` to "
                "force-honour zero3_shard=True."
            )

        if auto_force_persistent:
            if _ws_early > 1:
                LOG.info(
                    "ProTrain auto-mode: picking Mode A "
                    "(force_all_persistent=True). On PCIe Gen3 RTX 3090, "
                    "DDP+Mode_A gives ~3.6x scaling vs ZeRO-3's ~0.7x — see "
                    "DESIGN.md §Multi-GPU for benchmark data."
                )
            else:
                LOG.info(
                    "ProTrain auto-mode: picking Mode A "
                    "(force_all_persistent=True, single-rank)."
                )
        elif not auto_zero3:
            LOG.info(
                "ProTrain auto-mode: picking Mode B (CPU-offload, "
                "replicated). Per-rank CPU RAM sufficient for the full "
                "non-persistent chunk set."
            )
        else:
            LOG.info(
                "ProTrain auto-mode: picking Mode C (CPU-offload, "
                "ZeRO-3 sharded). Per-rank CPU RAM too tight for "
                "replication — falling back to 1/world_size shard."
            )

        force_all_persistent = auto_force_persistent
        zero3_shard = auto_zero3
        # Sync the downstream hardware_profile to the selector's pick.
        # The SEARCH ran with the most-permissive ``zero3_shard`` flag
        # (True on auto + multi-rank, see the resolve block above) so
        # the CPU gate didn't preempt Mode C. Now that the selector has
        # made its call, re-stamp the RUNTIME profile so the
        # chunk-manager, cost-model peak prediction, and any phase-2
        # rebuild see the ACTUAL mode the runtime will use (Mode B →
        # False, Mode C → True; Mode A → False because
        # force_all_persistent skips the sharded all_gather path).
        #
        # IMPORTANT: ``search_hw_profile`` (snapshot taken above
        # before this block) stays un-restamped — the phase-2
        # re-search MUST use that permissive profile. Otherwise the
        # stricter ``zero3_shard=False`` (e.g. when the selector
        # picked Mode A or Mode B) would re-engage the CPU
        # feasibility gate against the replicated footprint and
        # could filter out Mode-C-only candidates whose pinned CPU
        # only fits under sharding. The post-re-search
        # ``_select_mode`` call re-evaluates the runtime mode for
        # the post-measurement cfg.
        if zero3_shard != hardware_profile.zero3_shard:
            from dataclasses import replace as _replace

            hardware_profile = _replace(hardware_profile, zero3_shard=bool(zero3_shard))

    # ---- 4. construct runtime ------------------------------------------
    # When phase-2 is enabled (default on cache-miss profiles where the
    # backward was skipped), build under a CONSERVATIVE bootstrap config
    # first, take a chunked-runtime backward measurement, splice it into
    # the trace, persist, re-run search, and — if the new pick differs
    # from the bootstrap — tear down + rebuild under the post-research
    # cfg. The optimizer state slots are NOT yet wired into the trainer
    # at this point (the plugin's create_optimizer / post_trainer_create
    # pass haven't fired), so a rebuild here is safe.
    n_block = len(trace.activation_sizes)
    use_phase2 = (
        torch.cuda.is_available()
        and trace.steady_bwd_chunked_wall_s == 0.0
        and n_block > 0
        # Skip phase-2 calibration on the explicit-override and
        # force_all_persistent paths. Both paths have already
        # materialized a deterministic ``SearchResult`` from caller-
        # supplied knobs (see the ``force_all_persistent`` and
        # ``all_overrides_set`` branches above), and phase-2's post-
        # measurement re-search would silently replace that cfg with
        # the searcher's own pick — defeating the override (e.g. the
        # M5 OFFLOAD-mode tests would lose ``n_offload>0`` because
        # the searcher would re-pick a fits-on-GPU cfg with
        # ``n_offload=0``). Phase-2's whole point is to refine a
        # search-derived cfg with measured backward times; on the
        # explicit/forced paths there is nothing to refine.
        and not force_all_persistent
        and not all_overrides_set
    )
    if use_phase2:
        from axolotl.integrations.protrain.profiler.phase2 import (
            estimate_per_block_recompute_s,
            measure_chunked_steady,
            select_bootstrap_config,
        )

        boot_cfg, boot_block_map = select_bootstrap_config(
            initial_result=result,
            layout=layout,
            n_block=n_block,
            capacity_bytes=capacity_bytes,
            trace=trace,
            hw=hardware_profile,
        )
        boot_result = SearchResult(
            cfg=boot_cfg,
            block_map=boot_block_map,
            predicted_peak_bytes=result.predicted_peak_bytes,
            predicted_iter_s=result.predicted_iter_s,
        )
        chunk_manager, scheduler, handles, boot_result = _construct_runtime(
            model=model,
            blocks=blocks,
            layout=layout,
            result=boot_result,
            hardware_profile=hardware_profile,
            capacity_bytes=capacity_bytes,
            trace=trace,
            zero3_shard=zero3_shard,
            device=device,
        )
        # Build a transient WrappedModel + optimizer for the measurement.
        boot_wrapped = WrappedModel(
            module=model,
            search_result=boot_result,
            chunk_manager=chunk_manager,
            scheduler=scheduler,
            _hook_handles=list(handles),
        )
        from axolotl.integrations.protrain.api.optim_wrapper import (
            protrain_optimizer_wrapper,
        )

        boot_optim = protrain_optimizer_wrapper(boot_wrapped, lr=1e-4)
        boot_batch = _dummy_batch(model, batch_size, seq_len, device)

        measurement_failed = False
        fwd_s = 0.0
        bwd_s = 0.0
        step_s = 0.0
        phase2_peak_bytes = 0
        try:
            fwd_s, bwd_s, step_s, phase2_peak_bytes = measure_chunked_steady(
                model=model, batch=boot_batch, optimizer=boot_optim
            )
        except Exception as exc:  # noqa: BLE001 — measurement is best-effort
            exc_repr = f"{type(exc).__name__}: {exc}"
            LOG.warning(
                "Phase-2 chunked measurement raised %s; falling back to "
                "the v8 cost-model path under the searcher's original "
                "pick. Tighten or disable the phase-2 gate if the "
                "failure is reproducible.",
                exc_repr,
            )
            del exc
            measurement_failed = True

        if measurement_failed:
            # Tear down the bootstrap runtime and rebuild under the
            # original search's pick. Phase-2 must be transparent on
            # failure — callers should see the same wrapper behavior
            # they'd get with phase-2 disabled. Unwrap blocks so the
            # rebuild's _build_block_spans sees the original param
            # names that match layout.chunks (see the cfg-changed
            # teardown branch for the full explanation).
            for h in handles:
                try:
                    h.remove()  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001 — best-effort
                    LOG.debug(
                        "phase-2 fallback teardown: hook handle remove failed: %s",
                        exc,
                    )
            block_parent_map_unwrap = _find_block_parent_map(model, blocks)
            for idx, block in enumerate(blocks):
                unwrapped = unwrap_block(block)
                if unwrapped is not block:
                    parent = block_parent_map_unwrap.get(id(block))
                    if parent is not None:
                        for slot, child in enumerate(parent):
                            if child is block:
                                parent[slot] = unwrapped
                                break
                    blocks[idx] = unwrapped
            chunk_manager.restore_to_gpu()
            del boot_wrapped, boot_optim, chunk_manager, scheduler, handles
            chunk_manager, scheduler, handles, result = _construct_runtime(
                model=model,
                blocks=blocks,
                layout=layout,
                result=result,
                hardware_profile=hardware_profile,
                capacity_bytes=capacity_bytes,
                trace=trace,
                zero3_shard=zero3_shard,
                device=device,
            )
        if not measurement_failed:
            # ``estimate_per_block_recompute_s`` derives a per-block
            # recompute estimate from ``_fwd_compute_time_from_trace``.
            # For TRACE_VERSION 11 the per-op-derived per-block shape is
            # what the bwd-translation in ``_bwd_compute_time_from_trace``
            # consumes (both the bootstrap subtraction AND the per-cfg
            # add) — so it stays consistent regardless of whether we
            # call it pre- or post-splice. We call it pre-splice to
            # mirror the v10 ordering and keep the splice block compact.
            per_block_recompute_s = estimate_per_block_recompute_s(trace, n_block)
            from dataclasses import replace as _replace

            new_trace = _replace(
                trace,
                steady_fwd_chunked_wall_s=fwd_s,
                steady_bwd_chunked_wall_s=bwd_s,
                steady_step_overlap_s=step_s,
                steady_phase2_peak_bytes=phase2_peak_bytes,
                phase2_n_persist=boot_result.cfg.n_persist,
                phase2_n_buffer=boot_result.cfg.n_buffer,
                phase2_n_checkpoint=boot_result.cfg.n_checkpoint,
                phase2_per_block_recompute_s=per_block_recompute_s,
            )
            try:
                save_cached_trace(cache_key, new_trace, cache_dir=cache_dir)
            except OSError as exc:
                LOG.warning(
                    "Phase-2: failed to persist updated trace (%s); the "
                    "in-memory trace is still updated for this run.",
                    exc,
                )
            trace = new_trace

            # Re-run search with phase-2 fields populated. Reuse the
            # same CPU feasibility budget — phase-2 only refines runtime
            # estimates, not memory accounting, so the CPU envelope
            # binding doesn't change.
            #
            # Pass ``search_hw_profile`` (the permissive snapshot taken
            # before ``_select_mode`` re-stamped ``hardware_profile``).
            # If we passed the runtime-stamped profile, then on auto-
            # mode runs where the original selector picked Mode A or
            # Mode B (zero3_shard=False) the search's CPU feasibility
            # gate would re-engage against the replicated footprint
            # and could drop Mode-C-only candidates whose pinned CPU
            # only fits under sharding. The post-search ``_select_mode``
            # call below picks the actual runtime mode for the new cfg.
            new_result = search(
                trace,
                layout,
                capacity_bytes,
                search_hw_profile,
                cpu_capacity_bytes=cpu_capacity_bytes,
            )

            # Re-pick runtime mode for the post-measurement cfg. The
            # original ``_select_mode`` decision was made against
            # ``boot_cfg``; ``new_result.cfg`` may push more chunks to
            # CPU (offload mode B/C) or fewer (Mode A), changing the
            # required per-rank CPU footprint and therefore the
            # replicated-vs-sharded-vs-A decision. Skip on the non-
            # auto path — explicit user flags don't get re-evaluated.
            mode_changed = False
            if auto_mode:
                cpu_ram_re = _cpu_ram_per_rank_bytes(_ws_early)
                new_force_persistent, new_zero3 = _select_mode(
                    search_result=new_result,
                    layout=layout,
                    hw=search_hw_profile,
                    world_size=_ws_early,
                    cpu_ram_per_rank_bytes=cpu_ram_re,
                    auto_mode=True,
                    user_force_all_persistent=_user_force_all_persistent,
                    user_zero3_shard=_user_zero3_shard,
                )
                # Re-stamp the runtime ``hardware_profile`` to reflect
                # the post-measurement mode pick. A mode flip MUST
                # trigger the ``cfg_changed`` rebuild path below — even
                # when ``new_result.cfg`` and ``block_map`` match the
                # bootstrap pick, because the live ChunkManager was
                # constructed under the OLD mode and silently keeps
                # running under it (e.g. replicated CPU offload when
                # only sharded fits). Track ``mode_changed`` here and
                # fold it into ``cfg_changed`` so the no-rebuild
                # short-circuit can't strand us on the wrong runtime.
                mode_changed = (
                    new_force_persistent != force_all_persistent
                    or new_zero3 != zero3_shard
                )
                if mode_changed:
                    LOG.info(
                        "Phase-2: post-measurement _select_mode changed "
                        "the runtime mode (force_all_persistent %s -> %s, "
                        "zero3_shard %s -> %s); rebuilding the runtime.",
                        force_all_persistent,
                        new_force_persistent,
                        zero3_shard,
                        new_zero3,
                    )
                force_all_persistent = new_force_persistent
                zero3_shard = new_zero3
                if zero3_shard != hardware_profile.zero3_shard:
                    hardware_profile = _replace(
                        hardware_profile, zero3_shard=bool(zero3_shard)
                    )
            # Compare the SEARCH's raw pick (boot_cfg) against the
            # search's raw new pick (new_result.cfg) — NOT the
            # calibrated boot_result.cfg. The two used to diverge
            # because ``_construct_runtime`` widened ``cfg.n_persist``
            # to ``len(_persistent_ids)`` (the prefix ∪ non-block-chunk
            # pin set) post-calibration; that collapse has since been
            # removed (the augmented set is now plumbed through
            # ``layout.mandatory_persistent`` so the prefix is preserved
            # verbatim), but we keep the comparison against ``boot_cfg``
            # both for symmetry with the rest of the phase-2 flow and
            # to be robust against any future calibration knob that
            # might rewrite the cfg.
            #
            # ``mode_changed`` (set above on the auto path) also forces
            # a rebuild even when the cfg/block_map match — see the
            # ``mode_changed`` block above for rationale.
            cfg_changed = (
                new_result.cfg != boot_cfg
                or new_result.block_map != boot_block_map
                or mode_changed
            )
            if not cfg_changed:
                calibrated_peak = _calibrate_peak_with_actual_chunk_bytes(
                    original_peak=new_result.predicted_peak_bytes,
                    layout=layout,
                    chunk_manager=chunk_manager,
                    cfg=new_result.cfg,
                    trace=trace,
                    block_map=new_result.block_map,
                )
                if calibrated_peak != new_result.predicted_peak_bytes:
                    # Preserve the search's prefix — see the matching
                    # comment in ``_construct_runtime`` for why
                    # ``len(_persistent_ids)`` (the augmented set) is
                    # NOT a sound substitute for ``cfg.n_persist`` here.
                    new_result = SearchResult(
                        cfg=CostConfig(
                            n_persist=new_result.cfg.n_persist,
                            n_buffer=new_result.cfg.n_buffer,
                            n_swap=new_result.cfg.n_swap,
                            n_checkpoint=new_result.cfg.n_checkpoint,
                            # Option B: preserve n_offload through the
                            # phase-2 post-measurement calibration
                            # rebuild. Mirrors the same fix in the
                            # initial _construct_runtime calibration
                            # path above.
                            n_offload=new_result.cfg.n_offload,
                        ),
                        block_map=new_result.block_map,
                        predicted_peak_bytes=calibrated_peak,
                        predicted_iter_s=new_result.predicted_iter_s,
                    )
                LOG.info(
                    "Phase-2: post-measurement search picked the same cfg "
                    "(predicted_iter_s %.4f -> %.4f); keeping bootstrap "
                    "runtime in place.",
                    boot_result.predicted_iter_s,
                    new_result.predicted_iter_s,
                )
                result = new_result
                wrapped = boot_wrapped
                wrapped.search_result = result
            else:
                LOG.info(
                    "Phase-2: post-measurement search picked a different "
                    "cfg (%s -> %s); tearing down bootstrap runtime and "
                    "rebuilding under the new pick.",
                    boot_result.cfg,
                    new_result.cfg,
                )
                # Teardown: uninstall hooks, unwrap blocks (so the
                # rebuild's calibration sees the original parameter
                # names that match layout.chunks — wrap_block inserts a
                # ``.block.`` infix into named_parameters() paths which
                # would otherwise make _build_block_spans miss every
                # block param), restore params to standalone GPU
                # storage, drop the bootstrap chunk_manager. The next
                # _construct_runtime re-wraps under the new block_map
                # via wrap_block (which is itself idempotent).
                for h in handles:
                    try:
                        h.remove()  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        LOG.debug(
                            "phase-2 teardown: hook handle remove failed: %s",
                            exc,
                        )
                block_parent_map_unwrap = _find_block_parent_map(model, blocks)
                for idx, block in enumerate(blocks):
                    unwrapped = unwrap_block(block)
                    if unwrapped is not block:
                        parent = block_parent_map_unwrap.get(id(block))
                        if parent is not None:
                            for slot, child in enumerate(parent):
                                if child is block:
                                    parent[slot] = unwrapped
                                    break
                        blocks[idx] = unwrapped
                chunk_manager.restore_to_gpu()
                del boot_wrapped, boot_optim, chunk_manager, scheduler, handles
                chunk_manager, scheduler, handles, result = _construct_runtime(
                    model=model,
                    blocks=blocks,
                    layout=layout,
                    result=new_result,
                    hardware_profile=hardware_profile,
                    capacity_bytes=capacity_bytes,
                    trace=trace,
                    zero3_shard=zero3_shard,
                    device=device,
                )
    else:
        chunk_manager, scheduler, handles, result = _construct_runtime(
            model=model,
            blocks=blocks,
            layout=layout,
            result=result,
            hardware_profile=hardware_profile,
            capacity_bytes=capacity_bytes,
            trace=trace,
            zero3_shard=zero3_shard,
            device=device,
        )

    LOG.info(
        "ProTrain config: n_persist=%d n_buffer=%d n_swap=%d n_checkpoint=%d "
        "S_chunk=%d N_chunk=%d peak=%.2f GiB iter=%.3f s capacity=%.2f GiB",
        result.cfg.n_persist,
        result.cfg.n_buffer,
        result.cfg.n_swap,
        result.cfg.n_checkpoint,
        layout.S_chunk,
        layout.N_chunk,
        result.predicted_peak_bytes / (1 << 30),
        result.predicted_iter_s,
        capacity_bytes / (1 << 30),
    )

    wrapped = WrappedModel(
        module=model,
        search_result=result,
        chunk_manager=chunk_manager,
        scheduler=scheduler,
        _hook_handles=list(handles),
    )
    # Stash the searcher inputs so the plugin's post_trainer_create hook
    # can re-run search() once the distributed process group is up and
    # real NCCL collectives become measurable. The trace was profiled
    # before dist.init, so its nccl_gather_s / nccl_reduce_s tables are
    # empty whenever the wrapper runs from post_model_load with
    # world_size > 1 — see DESIGN.md "NCCL measurement gap".
    wrapped._trace = trace  # type: ignore[attr-defined]
    wrapped._layout = layout  # type: ignore[attr-defined]
    wrapped._capacity_bytes = int(capacity_bytes)  # type: ignore[attr-defined]
    # Carry the CPU feasibility budget through so the plugin's
    # post_trainer_create remeasure path can reuse the same hard filter
    # when it re-runs the search after dist init.
    wrapped._cpu_capacity_bytes = (  # type: ignore[attr-defined]
        int(cpu_capacity_bytes) if cpu_capacity_bytes is not None else None
    )
    wrapped._hardware_profile = hardware_profile  # type: ignore[attr-defined]
    wrapped._cache_key = cache_key  # type: ignore[attr-defined]
    # Carry the user-supplied cache_dir so post_trainer_create's NCCL
    # re-measure path can persist the spliced trace under the same root.
    wrapped._cache_dir = cache_dir  # type: ignore[attr-defined]
    return wrapped


def auto_wrap(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    *,
    capacity_bytes: int | None = None,
    cpu_capacity_bytes: int | None = None,
    cache_dir: str | None = None,
) -> WrappedModel:
    """Drop-in ProTrain wrapper with auto-derived ``HardwareProfile``.

    Construct a :class:`HardwareProfile` from live ``torch.cuda`` queries
    and call :func:`protrain_model_wrapper` with auto-mode enabled. This
    is the closest analogue to the paper's Figure 1 single-line API for
    users who aren't going through the Axolotl plugin path:

    .. code-block:: python

        from axolotl.integrations import protrain

        wrapped = protrain.auto_wrap(model, batch_size=4, seq_len=2048)
        optimizer = protrain.protrain_optimizer_wrapper(wrapped)

    Parameters
    ----------
    model : nn.Module
        Any standard ``nn.Module``. Must already be on the GPU device
        you want ProTrain to use (place via ``model.cuda()`` or
        ``model.to(device)`` first). ``auto_wrap`` does not move models
        — explicit is better than implicit for device placement.
    batch_size, seq_len : int
        Used for both the profiler invocation and the cache key.
    capacity_bytes : int | None
        GPU memory budget passed through to the searcher. ``None``
        defaults to ``hw.gpu_memory_bytes - 2 GiB`` headroom (resolved
        inside :func:`protrain_model_wrapper`).
    cpu_capacity_bytes : int | None
        Per-rank pinned CPU RAM budget the searcher should treat as a
        hard feasibility filter. ``None`` auto-derives from
        ``psutil.virtual_memory()`` inside the wrapper.
    cache_dir : str | None
        Override the profiler cache root. When provided, traces are
        stored / loaded under ``<cache_dir>/protrain/profiler``, taking
        precedence over ``XDG_CACHE_HOME``. ``None`` falls back to the
        XDG-style default.

    Returns
    -------
    WrappedModel
        Ready for standard PyTorch training-loop use:
        ``loss.backward(); optimizer.step()``.

    Raises
    ------
    RuntimeError
        If CUDA is not available, or ``model`` parameters live on CPU
        (auto_wrap does not move models).

    See Also
    --------
    protrain_model_wrapper : the lower-level wrapper exposed for callers
        who need fine-grained control over the
        :class:`HardwareProfile`, override knobs, or sharding mode
        (``zero3_shard``, ``force_all_persistent``, the four-tuple
        ``n_*_override`` debug knobs).
    """
    import torch

    from axolotl.integrations.protrain.api.hardware import build_hardware_profile

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ProTrain.auto_wrap requires CUDA; torch.cuda.is_available() is False. "
            "Construct a HardwareProfile manually and call protrain_model_wrapper "
            "directly for non-CUDA test harnesses."
        )

    # Refuse to silently accept a CPU-resident model. The wrapper itself
    # would later raise inside the profiler's tracker (which calls
    # ``torch.cuda.memory_stats(device)`` on the model's device) — surface
    # the contract here with a clearer message.
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = None
    if param_device is not None and param_device.type != "cuda":
        raise RuntimeError(
            f"ProTrain.auto_wrap requires the model to be on GPU before the call "
            f"(found device={param_device}). Move it first via "
            f"`model.cuda()` or `model.to('cuda:N')`, then re-invoke."
        )

    hw = build_hardware_profile()

    return protrain_model_wrapper(
        model,
        model_config=getattr(model, "config", None),
        hardware_profile=hw,
        batch_size=batch_size,
        seq_len=seq_len,
        capacity_bytes=capacity_bytes,
        cpu_capacity_bytes=cpu_capacity_bytes,
        cache_dir=cache_dir,
        auto_mode=True,
    )


def _find_block_parent_map(
    model: nn.Module, blocks: list[nn.Module]
) -> dict[int, "nn.ModuleList"]:
    """Map ``id(block)`` to the ``nn.ModuleList`` containing it.

    ``flatten_block_trees(discover_blocks(model))`` returns a plain
    ``list`` whose elements may live in **multiple** ``nn.ModuleList``
    instances (encoder.block + decoder.block on T5). To swap in wrapped
    modules we need each block's true parent so the in-place
    ``parent[slot] = wrapped`` reassignment propagates to the rest of
    the model.

    Walks every ``nn.ModuleList`` under ``model`` once and records the
    parent for every block's ``id()`` it sees. Blocks not found in any
    ``ModuleList`` (defensive — should not happen for blocks returned
    by ``discover_blocks``) are silently absent from the map; the
    wrap/unwrap path then leaves them in place.
    """
    out: dict[int, "nn.ModuleList"] = {}
    if not blocks:
        return out
    target_ids = {id(b) for b in blocks}
    for module in model.modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for child in module:
            cid = id(child)
            if cid in target_ids and cid not in out:
                out[cid] = module
    return out


__all__ = ["auto_wrap", "protrain_model_wrapper"]
