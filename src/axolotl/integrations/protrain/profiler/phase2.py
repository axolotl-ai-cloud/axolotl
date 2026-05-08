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
from typing import TYPE_CHECKING, Any

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
    """Minimum pool size needed for adjacent-block prefetch at ``n_persist``.

    The ``persistent`` set is the layout's *effective* persistent set —
    ``[0, n_persist)`` plus any ``layout.mandatory_persistent`` chunks
    (large-buffer / non-shardable chunks the runtime always keeps GPU-
    resident). Excluding only ``[0, n_persist)`` would over-count buffer
    needs for adjacency by counting mandatory-persistent chunks twice
    (once as resident, once as needing a slot), inflating the bootstrap
    peak and forcing phase-2 off the intended zero-persist/full-CKPT
    calibration path. ``ChunkLayout.effective_persistent_ids(n_persist)``
    is the canonical source of truth for this set.
    """
    if n_persist >= layout.N_chunk:
        return 0
    persistent: set[ChunkId] = set(layout.effective_persistent_ids(n_persist))
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


def _clone_state_dict(state, target_device=None):
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

    Mapping-type preservation: ``Module.state_dict()`` returns an
    ``OrderedDict`` subclass that carries a ``_metadata`` attribute
    (per-module version info, used by ``load_state_dict`` for
    backward-compat upgrades). A naive ``dict(...)`` rebuild strips
    both the subclass identity and ``_metadata``, so ``load_state_dict``
    on the snapshot loses version info and silently drops any
    upgrade-on-load path. We rebuild via ``type(state)(...)`` and
    shallow-copy ``_metadata`` if present.

    Parameters
    ----------
    state
        The state-dict (or nested element) to clone.
    target_device : torch.device | str | None
        When set and the leaf is a tensor, the snapshot tensor is
        relocated to ``target_device`` (``state.detach().to(target_device).clone()``)
        so the snapshot does not duplicate GPU memory for state we
        intend to keep on host. When ``None`` the snapshot preserves
        the source tensor's device (the default — model state must be
        snapshotted on-device to keep ``load_state_dict`` cheap).

        IMPORTANT for optimizer state: ``Optimizer.load_state_dict``
        casts the loaded per-parameter state tensors (e.g. Adam's
        ``exp_avg`` / ``exp_avg_sq``) onto the matching parameter's
        device automatically, so a CPU-resident snapshot is restorable
        — but the cast happens via a ``.to(param.device)`` copy at
        restore time, which means the GPU-side tensor is reallocated.
        Callers that need a faithful restore without the device-cast
        round-trip (e.g. tests asserting tensor identity) should pass
        the parameter device. Callers minimizing GPU memory during
        the timed region (the common case for ``measure_chunked_steady``)
        should pass ``torch.device("cpu")`` so the snapshot lives on
        host.
    """
    import torch

    if torch.is_tensor(state):
        if target_device is not None:
            target = torch.device(target_device)
            if state.device == target:
                # Already on the target device — single ``clone()``
                # allocates the snapshot directly without a needless
                # round-trip.
                return state.detach().clone()
            # ``.to(target)`` allocates a fresh tensor on the target
            # device when the source is on a different device, and
            # under that condition the result is independent storage —
            # no extra ``.clone()`` is needed (and the prior path
            # double-allocated, which on multi-GB model snapshots was
            # spiking host RAM enough to OOM phase-2 before timing
            # started).
            return state.detach().to(target)
        return state.detach().clone()
    if isinstance(state, dict):
        cloned_items = {
            k: _clone_state_dict(v, target_device=target_device)
            for k, v in state.items()
        }
        # Preserve dict subclass identity (e.g. OrderedDict). For plain
        # ``dict`` inputs ``type(state)(...)`` is equivalent to ``dict(...)``.
        try:
            cloned_dict = type(state)(cloned_items)
        except TypeError:
            # Fallback: subclass constructor doesn't accept a single
            # mapping arg (rare custom dicts). Rebuild as a plain dict.
            cloned_dict = dict(cloned_items)
        # ``_metadata`` is not part of the dict's items — it's set as a
        # plain attribute by ``Module.state_dict()`` and read back by
        # ``load_state_dict``. Shallow-copy preserves the per-module
        # version dict's identity contract without re-cloning tensors
        # (it doesn't contain any).
        metadata = getattr(state, "_metadata", None)
        if metadata is not None:
            try:
                cloned_dict._metadata = copy.copy(metadata)  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                # Some mapping subclasses reject arbitrary attribute
                # assignment — silently drop _metadata in that case
                # rather than crash the snapshot.
                pass
        return cloned_dict
    if isinstance(state, (list, tuple)):
        cloned = [_clone_state_dict(v, target_device=target_device) for v in state]
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

    # Capture caller-visible state BEFORE flipping into train mode so
    # the rollback at the end of the timed region restores the model
    # to exactly what the caller handed in: training-mode flag, CPU
    # RNG state, and CUDA RNG state on every visible device. The timed
    # loop calls ``model.train()`` and consumes random samples (e.g.
    # via dropout / data ordering), both of which would otherwise
    # leak into the caller's subsequent steps.
    # Snapshot per-module training flags BEFORE ``model.train()`` flips
    # them all. Without this, submodules that the caller had
    # individually placed in eval() — frozen LoRA backbones,
    # BatchNorm/Dropout in inference mode for partially-frozen
    # finetuning, MoE expert subsets, etc. — get incorrectly stuck in
    # train() after the function returns, because a top-level-only
    # rollback via ``model.train()`` / ``model.eval()`` recurses and
    # clobbers the previously-eval submodules. Keyed by ``id(m)`` so
    # we don't rely on module hashability. This snapshot supersedes
    # the older ``was_training = model.training`` single-flag capture:
    # the per-module pass already covers the root module.
    module_training: dict[int, bool] = {id(m): m.training for m in model.modules()}
    cpu_rng = torch.get_rng_state()
    cuda_rngs = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Bind every CUDA timing/memory API call to the model's device so a
    # future refactor that changes the current-device context between
    # plugin setup and measurement cannot silently measure the wrong GPU.
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Phase-2 measurement expected a CUDA model, got {device!r}")

    # Sentinels so the ``finally`` block knows which restore steps are
    # safe to run even when an exception fires partway through the
    # snapshot-and-warmup setup. Any operation that mutates caller-
    # visible state (``model.train()``, ``state_dict`` clones,
    # ``optimizer.zero_grad``) lives inside the ``try:`` so it can roll
    # back cleanly on partial failure.
    model_state: dict[str, Any] | None = None
    optim_state: dict[str, Any] | None = None
    # Set of state-dict keys intentionally filtered out of ``model_state``
    # at snapshot time (zero-element placeholder tensors that came from
    # offloaded chunks). The restore's ``strict=False`` accepts these
    # as missing; any OTHER missing key indicates that the timed region
    # mutated the model's state-dict surface in a way the snapshot
    # cannot roll back, which the restore promotes to a hard error.
    expected_missing_keys: frozenset[str] = frozenset()
    # Mode-C correctness gap: ``state_dict()`` only sees what's reachable
    # through ``param.data``. After the chunk manager's
    # ``materialize_offload`` runs, every non-persistent param's
    # ``param.data`` is an empty placeholder — the real weights live in
    # ``chunk_manager._cpu_slots`` (replicated) or
    # ``chunk_manager._chunk_shards`` (sharded). The optimizer step
    # below mutates those CPU bytes directly via the gather→step→offload
    # cycle, and ``load_state_dict(model_state)`` would no-op for those
    # weights — leaving the params permanently advanced past the
    # caller's pre-measurement state. Use the chunk manager's
    # snapshot/restore helpers when the optimizer is a ProTrain wrapper
    # (the only path where the chunk-manager rollback is needed); the
    # ``hasattr`` guard preserves stock-torch-optimizer compatibility.
    chunk_state: dict[Any, Any] | None = None
    chunk_manager = getattr(optimizer, "_chunk_manager", None)
    with torch.cuda.device(device):
        try:
            model.train()
            # Snapshot model + optimizer state BEFORE warmup so the
            # measurement (which calls ``optimizer.step()`` and
            # mutates parameters) is non-destructive: training
            # resumes from the same initial state after the profiler
            # returns. The snapshot itself is excluded from the
            # timed region — captured before warmup, restored after
            # the timed loop.
            #
            # ``state_dict()`` returns *aliased* tensor references —
            # the subsequent ``optimizer.step()`` calls would mutate
            # those tensors in-place and silently advance the
            # snapshot, so we deep-clone every tensor (independent
            # storage) before warmup. Synchronize first so any
            # in-flight kernels finish writing before we read
            # parameters / optimizer state into the clone.
            torch.cuda.synchronize(device)
            # Snapshot model state on host as well, mirroring the
            # optim-state path below — for a multi-GB model
            # ``state_dict()`` clones to GPU would double the
            # parameter footprint during the timed region.
            # ``Module.load_state_dict`` copies values into the live
            # parameters at restore time, so the saved CPU tensors
            # land back on each parameter's original device — no
            # device drift on rollback.
            #
            # Mode-C / PEFT correctness: filter out entries whose live
            # tensor is the zero-element offloaded placeholder
            # (``param.data = empty(0)`` after
            # ``ChunkManager.materialize_offload``). Including those
            # keys is doubly wrong:
            #
            #   * The placeholder carries no real bytes — the actual
            #     weights live in ``chunk_manager._cpu_slots`` /
            #     ``_chunk_shards`` and are restored via the
            #     ``chunk_state`` path below; snapshotting the
            #     placeholder ADDS NOTHING to the rollback contract.
            #   * On restore, ``Module.load_state_dict`` shape-checks
            #     each entry. A snapshot captured AT empty-placeholder
            #     time but a live param momentarily rebound to its
            #     real GPU buffer (e.g. the last timed step left a
            #     chunk gathered, or a PEFT-wrapped backbone has
            #     adapter modules whose ``state_dict`` walks see the
            #     gathered shape) raises
            #     ``RuntimeError: Error(s) in loading state_dict for
            #     PeftModelForCausalLM: size mismatch for ...``.
            #
            # Filtering here keeps the rollback minimal: only
            # GPU-resident persistent / LoRA-adapter / non-chunked
            # tensors round-trip through ``state_dict()`` (where they
            # belong); offloaded chunks round-trip via ``chunk_state``
            # (where their bytes actually live). Pair this with
            # ``strict=False`` on the restore so the skipped keys
            # don't trip ``load_state_dict``'s missing-keys check.
            #
            # ``v.numel() > 0`` is the placeholder filter:
            # ``_empty_placeholder`` returns ``torch.empty(0)`` exactly,
            # so a non-trivial parameter cannot collide with the
            # filter. Buffers whose source-of-truth is ``param.data``
            # already-empty (rare but valid — e.g. a deliberate empty
            # tensor in a checkpoint) are also skipped, but that's
            # benign: if the live tensor is empty before AND after
            # the timed loop, there is nothing to restore.
            full_state = model.state_dict()
            filtered_state = {
                k: v
                for k, v in full_state.items()
                if not torch.is_tensor(v) or v.numel() > 0
            }
            # Capture the set of keys we intentionally skipped (offloaded
            # placeholder tensors that ``_empty_placeholder`` materialised
            # as zero-element). The restore path uses ``strict=False`` to
            # tolerate them missing from the snapshot, so we need this
            # set to distinguish "expected missing" from "real missing"
            # below.
            expected_missing_keys = frozenset(full_state.keys() - filtered_state.keys())
            model_state = _clone_state_dict(
                filtered_state, target_device=torch.device("cpu")
            )
            # Snapshot optimizer state on host to avoid duplicating
            # GPU memory during the timed region — for FusedAdam-
            # style optimizers the per-param ``exp_avg`` /
            # ``exp_avg_sq`` tensors are the same size as the params
            # themselves, so an on-device snapshot doubles the
            # optimizer-state footprint.
            # ``Optimizer.load_state_dict`` casts the loaded state
            # back to each parameter's device at restore time, so a
            # CPU snapshot round-trips faithfully.
            #
            # ProTrain's ``_ProTrainOptimizer.state_dict`` is a hollow
            # ``{"state": {}, "param_groups": [...]}`` shell BY DESIGN
            # (CHECKPOINT_DESIGN.md §1.7 Option P): Accelerate's
            # ``prepare`` round-trips the snapshot through
            # ``move_to_device`` and a full snapshot would push every
            # CPU FusedAdam moment to GPU. The hollow shell makes the
            # public ``load_state_dict`` a no-op too — so the rollback
            # below would silently leak any moments the timed loop
            # mutated. Use the non-public snapshot/restore pair the
            # ProTrain wrapper exposes specifically for this consumer
            # (it walks the inner FusedAdam adapters directly). The
            # ``hasattr`` guard preserves stock-torch-optimizer
            # compatibility for non-ProTrain measurement paths.
            if hasattr(optimizer, "_protrain_snapshot_inner_state"):
                optim_state = _clone_state_dict(
                    optimizer._protrain_snapshot_inner_state(),
                    target_device=torch.device("cpu"),
                )
            else:
                optim_state = _clone_state_dict(
                    optimizer.state_dict(), target_device=torch.device("cpu")
                )
            # Snapshot the chunk manager's pinned-host param shadows for
            # every non-persistent chunk. The ``Module.state_dict()``
            # snapshot above misses these — see the comment by the
            # ``chunk_state`` declaration. ``snapshot_cpu_state`` returns
            # plain (unpinned) host clones, so the snapshot adds RAM
            # equal to the offloaded-chunk byte total but keeps GPU
            # untouched. ``getattr`` covers older managers that pre-date
            # the helper (none in tree, but defensive).
            if chunk_manager is not None and hasattr(
                chunk_manager, "snapshot_cpu_state"
            ):
                chunk_state = chunk_manager.snapshot_cpu_state()
            # Start from a clean grad state so leftover grads from
            # prior trace work (e.g. the phase-1 profile pass) cannot
            # pollute the first warmup step's peak-memory and timing
            # samples.
            optimizer.zero_grad(set_to_none=True)
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
            #
            # Each restore is gated on the matching snapshot being
            # populated: if e.g. ``_clone_state_dict(model.state_dict())``
            # itself raised, ``model_state`` is still ``None`` and the
            # mutating ``model.train()`` is the only state change we
            # need to roll back (handled by the per-module training-
            # flag restore at the bottom of this block).
            # Defer rollback errors so every restore step runs even if
            # one raises. Without this every later step (chunk_state,
            # optim_state, RNG, train/eval flags) would be skipped on
            # the first failure, leaving the caller with a partially
            # rolled-back model — strictly worse than the original
            # half-restored state we were trying to surface. Capture
            # the FIRST exception, finish all remaining restore steps,
            # and re-raise at the end of the ``finally`` block.
            restore_error: Exception | None = None
            torch.cuda.synchronize(device)
            if model_state is not None:
                # ``strict=False`` because the snapshot is intentionally
                # filtered — offloaded chunks' empty-placeholder
                # ``param.data`` entries are skipped at snapshot time
                # (their real bytes round-trip via ``chunk_state``).
                # ``Module.load_state_dict(strict=False)`` returns a
                # ``_IncompatibleKeys`` namedtuple of
                # ``(missing_keys, unexpected_keys)``; missing keys
                # here are by-construction the offloaded set and are
                # NOT a bug, but unexpected keys WOULD signal a real
                # contract violation (the live model gained keys
                # during measurement) — surface those as a debug log
                # so a future regression is visible in trace output
                # without crashing the measurement.
                try:
                    _result = model.load_state_dict(model_state, strict=False)
                    # Validate ``missing_keys`` against the
                    # snapshot-time skipped set. ``strict=False`` is the
                    # right escape hatch for the offloaded zero-sized
                    # placeholders we filtered out at snapshot, but any
                    # OTHER missing key means the timed region mutated
                    # the model's state-dict surface (added a parameter,
                    # rebuilt a submodule, etc.) in a way the snapshot
                    # cannot restore. Surface that as a hard error so a
                    # partial restore doesn't silently leave the model
                    # in a half-rolled-back state.
                    extra_missing = set(_result.missing_keys) - expected_missing_keys
                    if extra_missing:
                        raise RuntimeError(
                            "Phase-2 state_dict restore missed "
                            f"{len(extra_missing)} unexpected keys "
                            f"(first 3: {sorted(extra_missing)[:3]}). "
                            "The live model's state-dict surface "
                            "changed during the timed measurement; "
                            "investigate the harness or the model for "
                            "a parameter add / submodule rebuild."
                        )
                    if _result.unexpected_keys:
                        # ``unexpected_keys`` = keys present in the snapshot
                        # but NOT in the live model — the live model dropped
                        # or renamed state during the timed measurement.
                        # Those snapshot bytes therefore did NOT make it
                        # back, so the rollback is incomplete and the
                        # caller is left with a mutated model. Promote to
                        # a hard error to match the ``extra_missing``
                        # symmetry above (both directions of state-dict
                        # surface drift now fail loudly rather than
                        # silently leaving a half-restored model).
                        raise RuntimeError(
                            "Phase-2 state_dict restore saw "
                            f"{len(_result.unexpected_keys)} unexpected snapshot "
                            f"keys (first 3: {_result.unexpected_keys[:3]}). "
                            "The live model dropped or renamed state during "
                            "the timed measurement, so rollback is incomplete."
                        )
                except Exception as exc:  # noqa: BLE001 — re-raised below
                    restore_error = restore_error or exc
            # Restore the chunk-manager CPU-shadow bytes BEFORE the
            # optimizer state restore. Order doesn't strictly matter
            # for correctness — the two snapshots cover disjoint
            # storage (model weights vs. Adam moments) — but keeping
            # the parameter rollback (state_dict + chunk_state)
            # adjacent makes the data-flow pairing easier to read.
            # Skip cleanly when the manager isn't a ProTrain wrapper
            # or the snapshot was never taken (chunk_state None).
            if (
                chunk_state is not None
                and chunk_manager is not None
                and hasattr(chunk_manager, "restore_cpu_state")
            ):
                try:
                    chunk_manager.restore_cpu_state(chunk_state)
                except Exception as exc:  # noqa: BLE001 — re-raised below
                    restore_error = restore_error or exc
            if optim_state is not None:
                # Mirror the snapshot path: route through the
                # ProTrain non-public restore helper when present so
                # the inner FusedAdam adapters actually receive the
                # moments back (the public ``load_state_dict`` is a
                # no-op by design — see the snapshot block above).
                try:
                    if hasattr(optimizer, "_protrain_restore_inner_state"):
                        optimizer._protrain_restore_inner_state(optim_state)
                    else:
                        optimizer.load_state_dict(optim_state)
                    optimizer.zero_grad(set_to_none=True)
                except Exception as exc:  # noqa: BLE001 — re-raised below
                    restore_error = restore_error or exc
            torch.cuda.synchronize(device)
            # Restore RNG state + training flag AFTER the parameter /
            # optimizer rollback so the caller observes byte-identical
            # state to what they handed in. Order matters: the
            # state_dict restore above can run kernels that consume
            # RNG, so RNG must be restored last.
            torch.set_rng_state(cpu_rng)
            if cuda_rngs is not None:
                torch.cuda.set_rng_state_all(cuda_rngs)
            # Restore per-module training flags AFTER the state_dict +
            # RNG rollback so the module-level state lands last and
            # nothing the rollback runs (e.g. autograd kernels invoked
            # by ``load_state_dict``) can re-flip flags. This is the
            # canonical restore — it covers the top-level module and
            # every submodule independently, so the caller observes
            # byte-identical mode state to what they handed in
            # (frozen-eval submodules stay eval, etc.).
            for m in model.modules():
                saved = module_training.get(id(m))
                if saved is None:
                    # New module attached during measurement (vanishingly
                    # rare — would imply the timed region rebuilt the
                    # graph). Leave whatever ``model.train()`` set it to.
                    continue
                if saved:
                    m.train()
                else:
                    m.eval()
            # All restore steps have run. Re-raise the first captured
            # error (if any). Re-raising AFTER all restore steps means
            # the caller observes the partial-restore failure but the
            # other restore directions (chunk_state, optim_state, RNG,
            # train/eval flags) still completed — strictly better than
            # raising mid-rollback and leaving the unrestored steps
            # silently mutated.
            if restore_error is not None:
                raise restore_error
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
    t_fwd_total, per_block_compute, _used_measured, _fwd_compute_base = (
        _fwd_compute_time_from_trace(trace)
    )
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
    "estimate_per_block_recompute_s",
    "measure_chunked_steady",
    "select_bootstrap_config",
]
