"""Public optimizer-wrapper for the ProTrain runtime (§1, §5).

``protrain_optimizer_wrapper`` returns a :class:`torch.optim.Optimizer`
subclass that proxies ``step`` / ``zero_grad`` through the persistent
(GPU FusedAdam) and non-persistent (CPU FusedAdam, async) adapters
already instantiated by :func:`protrain_model_wrapper`.

Semantics:

* ``step()`` — synchronously runs the GPU step for persistent chunks,
  then blocks on every outstanding CPU Adam future so the non-persistent
  chunk updates have landed in their CPU shards before control returns.
* ``zero_grad()`` — zeros grads on both adapters.
* ``state_dict`` — returns a hollow shell tagged with a
  ``_protrain_hollow_state_dict`` marker, preserving the
  ``{"state": ..., "param_groups": ...}`` shape HF Trainer +
  Accelerate expect at ``prepare`` time. Real adapter moments are
  persisted via the dedicated ProTrain checkpoint hook
  (``api/checkpoint.py``), NOT through this method.
* ``load_state_dict`` — silently no-ops when fed back the hollow
  shell (Accelerate prepare round-trip, or a user
  ``torch.save(state_dict()) → torch.load`` over the same wrapper).
  Raises ``NotImplementedError`` on any OTHER payload (e.g.
  state from a stock optimizer the user wants to migrate from), with
  a pointer at the ProTrain checkpoint hook. This is option (b) from
  ``CHECKPOINT_DESIGN.md`` §1.7 — the explicit-error variant chosen
  to close the silent-no-op footgun for direct ``auto_wrap`` users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from axolotl.integrations.protrain.chunk import (
    CpuFusedAdamAdapter,
    GpuAdamW8bitAdapter,
    GpuFusedAdamAdapter,
)
from axolotl.integrations.protrain.types import ChunkId, WrappedModel
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

    from axolotl.integrations.protrain.chunk import ChunkManager

LOG = get_logger(__name__)


class _ProTrainOptimizer(torch.optim.Optimizer):
    """``torch.optim.Optimizer`` facade over the ProTrain adapter pair.

    We inherit from ``torch.optim.Optimizer`` primarily for interface
    compatibility with HuggingFace Trainer (which calls
    ``isinstance(optim, torch.optim.Optimizer)``); the actual update
    math is delegated to the two adapters.
    """

    def __init__(
        self,
        gpu_optim: GpuFusedAdamAdapter | GpuAdamW8bitAdapter | None,
        cpu_optim: CpuFusedAdamAdapter | None,
        params: list["nn.Parameter"],
        defaults: dict[str, Any],
        chunk_manager: Any,
    ) -> None:
        """Wire the GPU/CPU adapter pair into a Trainer-compatible Optimizer facade."""
        # ``torch.optim.Optimizer.__init__`` requires at least one non-empty
        # parameter group. We pass the full param list so ``optim.param_groups``
        # reflects the real set — schedulers iterating over it still see
        # every tuneable param. The base class uses these only for
        # ``load_state_dict`` bookkeeping; the actual updates are routed
        # through the adapters in ``step``.
        if not params:
            # An empty-param optimizer is nonsensical — but during some smoke
            # tests every chunk can end up persistent and cpu_optim can be
            # None; we still need ``Optimizer`` super-init to succeed. Seed
            # with a dummy zero tensor in that case (torch rejects an empty
            # param group).
            raise ValueError(
                "_ProTrainOptimizer: model has no tunable parameters; "
                "nothing to optimize."
            )
        super().__init__(params, defaults)
        self._gpu_optim = gpu_optim
        self._cpu_optim = cpu_optim
        self._chunk_manager = chunk_manager

    # ---- step / zero_grad ----------------------------------------------

    def step(self, closure: Any = None) -> Any:
        """Drive both adapters then block on in-flight CPU futures.

        Persistent chunks: run the GPU step synchronously.
        Non-persistent chunks: per-param post-accumulate-grad hooks
        (installed by :meth:`ChunkManager.materialize_offload`) already
        kicked off the CPU FusedAdam step the instant each chunk's last
        grad landed on CPU — except in the **sharded** path
        (``zero3_shard=True``), where the per-param hook is intentionally
        a counter-only no-op and the chunk-level reduce_scatter +
        CPU-Adam kick lives in :meth:`reduce_grads_and_offload`, which
        the block-backward hook fires through
        :meth:`Scheduler.post_block_backward`.

        Block-backward hooks only attach to modules discovered as
        transformer blocks. Chunks owned by **non-block** modules
        (top-level ``lm_head`` / ``embed_tokens`` on a ``LlamaForCausalLM``,
        anything outside the decoder layer stack) therefore have no
        hook driving their ``reduce_grads_and_offload`` call — in the
        sharded path that means their grads sit unscattered, the CPU
        Adam step never fires, and those params silently DON'T update
        across iterations. Empirically this is enough to flatline the
        M6 Mode-C loss curve (the lm_head dominates the iter-1 logits
        and never leaves its init).

        Fix: before we wait on the CPU futures, sweep every
        non-persistent chunk and call ``reduce_grads_and_offload`` on
        it. The call is idempotent — chunks already processed by a
        block-backward hook find no live ``param.grad`` and early-return
        out of ``_reduce_scatter_and_offload_shard`` without re-issuing
        the collective; chunks whose block-backward hook never fired
        (the lm_head / embed-tokens orphans above) get their reduce_scatter
        + CPU-Adam kick HERE, then the wait_cpu_optim_all() below drains
        them in the same window as the block-driven kicks.

        Closure handling: ``torch.optim.Optimizer.step`` permits an
        optional ``closure`` callable that re-evaluates the model and
        returns the loss; per the PyTorch contract we call it under
        :func:`torch.enable_grad` and return its result so LBFGS-style
        optimizers / Trainer paths that pass a closure are not silently
        broken. The replicated/offload paths rely on per-param
        ``register_post_accumulate_grad_hook`` for the CPU-Adam kick, so
        the orphan sweep below is gated on ``zero3_shard`` — running it
        unconditionally would burn a no-op pass over every non-persistent
        chunk on every step in the non-sharded paths.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Forward LR-scheduler-driven hyperparam updates from the facade's
        # public ``param_groups`` (which torch ``LRScheduler`` mutates) into
        # the inner adapters' wrapped torch ``Optimizer.param_groups`` —
        # they are independent dict objects and would otherwise see stale
        # values. We only copy keys that already exist on each inner
        # group so we never invent hyperparams the inner optim doesn't
        # understand. The facade has exactly one outer group with the
        # full param list (see ``__init__``), so we use group 0 as the
        # source of truth. ``GpuFusedAdamAdapter`` exposes its inner via
        # ``._optim`` (single optimizer); ``CpuFusedAdamAdapter`` keeps
        # one inner per chunk under ``._optims``.
        self._forward_hyperparams_to_inner_optims()

        # Orphan sweep: ensure every non-persistent chunk has been
        # reduced+offloaded before we wait. See the docstring above for
        # why this is necessary in the sharded path. Replicated/offload
        # paths rely on per-param ``register_post_accumulate_grad_hook``
        # so the sweep is unnecessary there.
        cm = self._chunk_manager
        if getattr(cm, "zero3_shard", False):
            non_persist = getattr(cm, "_non_persistent_ids", None)
            if non_persist:
                for cid in list(non_persist):
                    cm.reduce_grads_and_offload(cid)

        if self._gpu_optim is not None:
            self._gpu_optim.step()
        # Drain every in-flight CPU Adam future (M4.5 Gap 2: per-param
        # grad offload enqueued these from the grad hooks; the orphan
        # sweep above enqueued the rest).
        self._chunk_manager.wait_cpu_optim_all()
        return loss

    # ---- LR-scheduler hyperparam forwarding -----------------------------

    # ``weight_decay`` is intentionally NOT forwarded:
    # ``_split_optim_param_groups`` builds two inner param groups for
    # the GPU/CPU adapters — the regular group with the user's
    # ``weight_decay``, and a no-decay group with ``weight_decay=0``
    # for bias / LayerNorm-family params (mirrors HF Trainer's
    # ``get_decay_parameter_names`` semantics). Forwarding the
    # facade's single ``weight_decay`` would clobber the no-decay
    # group's 0 and apply weight decay to bias / norm params,
    # changing training behavior. If a future scheduler needs to
    # vary weight decay, it must thread per-inner-group values
    # through; the facade-level wd is not a valid source for the
    # multi-group case.
    _FORWARDED_HYPERPARAM_KEYS = ("lr", "betas", "eps")

    def _forward_hyperparams_to_inner_optims(self) -> None:
        """Copy facade ``param_groups[0]`` hyperparams to each inner optim.

        ``torch.optim.lr_scheduler.LRScheduler.step()`` mutates
        ``self.param_groups[i]['lr']`` (and Adam-family schedulers may
        also touch ``betas`` / ``eps``) on the outer facade. The inner
        adapter optimizers (``self._gpu_optim._optim`` and each entry
        in ``self._cpu_optim._optims``) hold their own ``param_groups``
        list of dicts and never see those mutations, so without this
        forwarding step their ``step()`` keeps using the construction-
        time LR forever. Defensive: we only write keys that already
        exist on the inner group dict so this never invents new fields
        the inner optim's update math doesn't read. ``weight_decay`` is
        explicitly excluded from the forwarded set — see the
        ``_FORWARDED_HYPERPARAM_KEYS`` comment above.
        """
        if not self.param_groups:
            return
        src = self.param_groups[0]

        def _push(inner_optim) -> None:
            if inner_optim is None:
                return
            for inner_group in inner_optim.param_groups:
                for key in self._FORWARDED_HYPERPARAM_KEYS:
                    if key in src and key in inner_group:
                        inner_group[key] = src[key]

        if self._gpu_optim is not None:
            _push(getattr(self._gpu_optim, "_optim", None))
        if self._cpu_optim is not None:
            for inner in getattr(self._cpu_optim, "_optims", {}).values():
                _push(inner)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        """Zero gradients on every adapter and any unrouted param-group entries."""
        if self._gpu_optim is not None:
            self._gpu_optim.zero_grad(set_to_none=set_to_none)
        if self._cpu_optim is not None:
            self._cpu_optim.zero_grad(set_to_none=set_to_none)
        # Also zero any param grads that weren't routed through either
        # adapter (e.g. buffers that slipped through the chunk layout) so
        # the next iteration starts clean.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    # ---- checkpointing: torch-side no-ops, real save/load lives in the
    # ProTrain checkpoint callback (M5/M6) -------------------------------
    #
    # ``protrain_optimizer_wrapper`` is exported in the public API and
    # ``create_optimizer`` returns the raw wrapper before
    # ``post_trainer_create`` would have a chance to monkey-patch the
    # instance. HF Trainer (when ``save_only_model`` is False) and
    # Accelerate (at ``prepare`` time, unconditionally) both call
    # ``state_dict`` / ``load_state_dict`` on the optimizer; raising
    # ``NotImplementedError`` here would crash any out-of-trainer
    # consumer (model_wrapper.py profiling, tests). The adapters own
    # their own state and persist it through the dedicated ProTrain
    # checkpoint hook, so torch-side state is safely empty.

    #: Sentinel key marking a state_dict produced by this class as a
    #: hollow shell (CHECKPOINT_DESIGN.md §1.7 Option P). Lets
    #: ``load_state_dict`` distinguish the safe round-trip case
    #: (Accelerate ``prepare`` walk OR user ``torch.save(state_dict()) →
    #: torch.load → load_state_dict``) from a payload from a different
    #: optimizer that was incorrectly fed to this wrapper. The marker
    #: is a plain bool so Accelerate's ``move_to_device`` / ``.to(...)``
    #: walks ignore it.
    _PROTRAIN_HOLLOW_MARKER_KEY = "_protrain_hollow_state_dict"

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return an empty torch-side optimizer state.

        Real ProTrain optimizer state (per-shard moments held inside the
        CPU/GPU FusedAdam adapters) is saved by the dedicated checkpoint
        callback (see ``api/checkpoint.py``), NOT through this method.
        We still preserve HF's ``{"state": ..., "param_groups": ...}``
        shape so Accelerate's ``move_to_device(state_dict, ...)`` +
        ``load_state_dict`` round trip at ``prepare`` time does not
        crash. A ``_protrain_hollow_state_dict: True`` marker is added
        so ``load_state_dict`` can recognise the round trip and silently
        no-op (instead of raising on payloads it can't actually
        consume).

        IMPORTANT: this method does NOT serialise adapter moments. A
        naive ``torch.save(optim.state_dict())`` / ``torch.load`` /
        ``optim.load_state_dict(...)`` round trip will discard
        per-parameter moments — the saved blob is the hollow shell.
        Use the ProTrain checkpoint flow
        (``_save_protrain_optim_dir`` / ``_load_protrain_optim_dir``,
        wired via the ``post_trainer_create`` hook) for real persistence.
        """
        next_param_idx = 0
        param_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            n_params = len(group["params"])
            param_groups.append(
                {k: v for k, v in group.items() if k != "params"}
                | {"params": list(range(next_param_idx, next_param_idx + n_params))}
            )
            next_param_idx += n_params
        return {
            self._PROTRAIN_HOLLOW_MARKER_KEY: True,
            "state": {},
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Round-trip the hollow shell or fail loudly on foreign payloads.

        Accepts the hollow shell produced by ``state_dict()`` (Accelerate
        ``prepare`` round-trip OR a user ``torch.save / torch.load``
        sequence over the same wrapper) — that path silently no-ops,
        consistent with the wrapper's documented contract that real
        persistence flows through the dedicated ProTrain checkpoint
        hook (``api/checkpoint.py``).

        Any OTHER payload — e.g. a state_dict produced by a different
        optimizer, or a real torch state_dict the user thinks should
        restore — raises ``NotImplementedError`` with a pointer at
        the dedicated checkpoint hook. Replacing the previous silent
        no-op stops the footgun where users assumed
        ``optim.load_state_dict(saved_blob)`` would restore moments.
        """
        if not isinstance(state_dict, dict):
            raise NotImplementedError(
                "_ProTrainOptimizer.load_state_dict requires a dict; got "
                f"{type(state_dict).__name__}. Use the ProTrain checkpoint "
                "hook (api/checkpoint.py::_load_protrain_optim_dir) for real "
                "optimizer-state restore."
            )
        if state_dict.get(
            self._PROTRAIN_HOLLOW_MARKER_KEY
        ) is True and not state_dict.get("state"):
            # Hollow shell round-trip — Accelerate prepare path or
            # user ``torch.save(state_dict()) → torch.load →
            # load_state_dict`` over the same wrapper. The shell has
            # nothing to restore by construction; silent no-op is
            # correct.
            return None
        raise NotImplementedError(
            "_ProTrainOptimizer.load_state_dict cannot restore an arbitrary "
            "torch optimizer state. The wrapper's public state_dict is a "
            "hollow shell by design (CHECKPOINT_DESIGN.md §1.7 Option P) — "
            "real per-shard FusedAdam moments are persisted via the "
            "dedicated ProTrain checkpoint flow. Load via "
            "api/checkpoint.py::_load_protrain_optim_dir (wired through "
            "Trainer._load_optimizer_and_scheduler in post_trainer_create) "
            "instead of torch.save / torch.load over state_dict."
        )

    # ---- non-public snapshot of the REAL inner adapter state ------------
    #
    # The public ``state_dict``/``load_state_dict`` above are intentionally
    # hollow: HF Trainer + Accelerate's ``prepare`` round-trips the
    # optimizer state through ``move_to_device(state_dict, gpu)``, and a
    # full snapshot would silently push every CPU FusedAdam moment tensor
    # to the GPU, ballooning HBM (CHECKPOINT_DESIGN.md §1.7 Option P).
    #
    # Phase-2 measurement also needs to roll back optimizer state across a
    # timed loop, but it neither serializes nor moves devices — it just
    # captures and restores the moments on the same process. We expose a
    # private snapshot/restore pair for that consumer; callers that don't
    # know about it (Accelerate, Trainer) keep getting the hollow shell.
    def _protrain_snapshot_inner_state(self) -> dict[str, Any]:
        """Snapshot the REAL inner adapter state (not the hollow public shell).

        Captures the underlying ``torch.optim.Optimizer.state_dict()`` of
        each adapter's wrapped optimizer:

        * ``"gpu"`` — ``self._gpu_optim._optim.state_dict()`` for the
          persistent-chunk FusedAdam, or ``None`` when there is no GPU
          adapter (or its inner optim is itself ``None`` for an empty
          persistent set).
        * ``"cpu_per_chunk"`` — mapping ``ChunkId -> state_dict`` for each
          per-chunk DeepSpeedCPUAdam in ``self._cpu_optim._optims``.

        Intended for the phase-2 profiler's snapshot-and-rollback path
        (see ``profiler/phase2.py::measure_chunked_steady``); the public
        ``state_dict`` MUST stay hollow for the Accelerate
        ``move_to_device`` round-trip.
        """
        gpu_state: dict[str, Any] | None = None
        if self._gpu_optim is not None:
            inner = self._gpu_optim._optim
            if inner is not None:
                gpu_state = inner.state_dict()
        cpu_state_per_chunk: dict[ChunkId, dict[str, Any]] = {}
        if self._cpu_optim is not None:
            for cid, inner in self._cpu_optim._optims.items():
                cpu_state_per_chunk[cid] = inner.state_dict()
        return {"gpu": gpu_state, "cpu_per_chunk": cpu_state_per_chunk}

    def _protrain_restore_inner_state(self, snapshot: dict[str, Any]) -> None:
        """Restore inner-adapter state previously captured by the snapshot helper.

        Companion to :meth:`_protrain_snapshot_inner_state`. Calls
        ``load_state_dict`` on each inner ``torch.optim.Optimizer`` —
        unlike the public ``load_state_dict`` (which is a no-op so
        Accelerate's ``prepare``-time payload round-trips harmlessly),
        this routes the captured moments back into the adapters that
        actually own training state.
        """
        gpu_state = snapshot.get("gpu")
        if (
            gpu_state is not None
            and self._gpu_optim is not None
            and self._gpu_optim._optim is not None
        ):
            self._gpu_optim._optim.load_state_dict(gpu_state)
        cpu_state_per_chunk = snapshot.get("cpu_per_chunk") or {}
        if self._cpu_optim is not None and cpu_state_per_chunk:
            for cid, inner in self._cpu_optim._optims.items():
                inner_state = cpu_state_per_chunk.get(cid)
                if inner_state is not None:
                    inner.load_state_dict(inner_state)


# HF Trainer's ``get_decay_parameter_names`` excludes bias and norm-layer
# parameters from weight decay by default; if we collapse everything into
# a single global ``weight_decay`` here we silently change training behavior
# relative to the stock Trainer path. HF's actual filter checks the parent
# MODULE type (``nn.LayerNorm`` and any class whose name contains "Norm" —
# RMSNorm, GroupNorm, etc.) and the parameter-name suffix ``"bias"``. Pure
# name-token matching (the previous implementation) silently missed
# LayerNorm-style weights that don't carry a ``norm`` infix in their dotted
# parameter name (e.g. ``ln_1.weight`` on a GPT-2 block, where ``ln_1`` is
# an ``nn.LayerNorm`` but the parameter name itself contains no "norm"
# token).


def _collect_no_decay_param_ids(module: "nn.Module") -> set[int]:
    """Return ``id(p)`` for every parameter HF Trainer would put in the no-decay group.

    Mirrors :meth:`transformers.Trainer.get_decay_parameter_names` —
    excluding parameters whose parent module is :class:`nn.LayerNorm`
    (or any class with ``Norm`` in its name, case-insensitive — RMSNorm,
    GroupNorm, etc.) OR whose parameter name ends with ``"bias"`` (case-
    insensitive). We do NOT depend on
    :func:`transformers.trainer_pt_utils.get_decay_parameter_names` —
    that symbol is not stably exposed across HF versions (it lives on
    the :class:`Trainer` instance), and a hard import would couple this
    file to a private import path. The module-walk below produces the
    same set without that dependency.
    """
    from torch import nn

    no_decay: set[int] = set()
    for mod_name, mod in module.named_modules():
        is_norm_module = (
            isinstance(mod, nn.LayerNorm) or "norm" in type(mod).__name__.lower()
        )
        for param_name, param in mod.named_parameters(recurse=False):
            full_name = f"{mod_name}.{param_name}" if mod_name else param_name
            if is_norm_module or full_name.lower().endswith("bias"):
                no_decay.add(id(param))
    return no_decay


def _collect_sharded_no_decay_shard_param_ids(
    chunk_manager: "ChunkManager",
    cpu_params_per_chunk: "dict[ChunkId, list[nn.Parameter]]",
    no_decay_orig_param_ids: set[int],
) -> set[int]:
    """Map the original-param no-decay set onto sharded ``shard_param`` ids.

    In the M7 sharded path each chunk's CPU FusedAdam steps over the
    flat per-region :class:`_DtypeRegion.shard_param` tensors rather
    than the original ``nn.Parameter`` objects. The no-decay set
    collected from ``module.named_parameters()`` is keyed by the
    original-param ``id()``, so a direct id-match on the shard_params
    finds nothing — and norm/bias params silently inherit the global
    ``weight_decay`` (CR PR #17 R3190973417).

    Strategy: for each sharded chunk we already have the byte layout in
    ``chunk_manager._cpu_slots[cid]`` (each slot carries ``param_id`` +
    ``byte_offset`` + ``numel * element_size``) and the per-region
    ``[chunk_offset, chunk_offset + region_bytes)`` extent. A region
    inherits no-decay status iff ANY source param whose byte range
    intersects the region is in the original no-decay set. This is the
    correctness-conservative direction: HF Trainer also drops the whole
    norm/bias param into the wd=0 group, so we never under-decay any
    source param that the upstream Trainer would have decayed; we may
    over-cover a few decay-bytes that share a region with a norm scale,
    but those bytes are the SAME bytes Mode-C would already keep at
    fp32 (and which dtype-splitting tends to put in their own region
    anyway).

    Granularity trade-off (CR PR #17 round-2): a strictly HF-equivalent
    fix would split each region into byte-precise decay / no-decay
    sub-extents and emit a separate optimizer entry per sub-extent.
    That requires synthesizing per-intersection slice views of
    ``region.shard_param``, registering each as its own
    ``nn.Parameter`` on the underlying FusedAdam, partitioning the
    region's gradient buffer to match, and tracking distinct optimizer
    state (``exp_avg`` / ``exp_avg_sq``) per sub-extent — a substantial
    refactor of the per-region CPU-Adam interface and a perf risk on
    the hot offload step. The over-cover surface is bounded in
    practice because Mode-C's dtype-driven region split typically
    isolates fp32 norm scales into their own region (no adjacent
    decay-class weights to over-cover), so the residual decay leakage
    is small. We keep the region-level mapping for v1 and revisit if
    measured divergence from HF Trainer warrants the refactor.

    Returns a set of ``id(shard_param)`` that should be treated as
    no-decay. Empty when the chunk manager has no sharded chunks
    populated, or when the no-decay source set is itself empty.
    """
    if not no_decay_orig_param_ids:
        return set()
    chunk_shards = getattr(chunk_manager, "_chunk_shards", None)
    if not chunk_shards:
        return set()
    cpu_slots_by_cid = getattr(chunk_manager, "_cpu_slots", {}) or {}
    no_decay_shard_ids: set[int] = set()
    for cid, _params in cpu_params_per_chunk.items():
        shard_state = chunk_shards.get(cid)
        if shard_state is None or not shard_state.regions:
            continue
        slots = cpu_slots_by_cid.get(cid, [])
        if not slots:
            continue
        # Pre-resolve each slot to (start, end, is_no_decay) once.
        slot_extents: list[tuple[int, int, bool]] = []
        for slot in slots:
            param = chunk_manager._params_by_id.get(slot.param_id)
            if param is None:
                continue
            start = int(slot.byte_offset)
            end = start + int(slot.numel) * int(slot.element_size)
            slot_extents.append((start, end, id(param) in no_decay_orig_param_ids))
        for region in shard_state.regions:
            r_start = int(region.chunk_offset)
            r_end = r_start + int(region.region_bytes)
            region_has_no_decay = False
            for s_start, s_end, slot_no_decay in slot_extents:
                if not slot_no_decay:
                    continue
                # Intersection check.
                if s_start < r_end and s_end > r_start:
                    region_has_no_decay = True
                    break
            if region_has_no_decay:
                no_decay_shard_ids.add(id(region.shard_param))
    return no_decay_shard_ids


def _split_optim_param_groups(
    inner: torch.optim.Optimizer | None,
    no_decay_param_ids: set[int],
) -> None:
    """Split each of ``inner.param_groups`` into a decay/no-decay pair in place.

    ``CpuFusedAdamAdapter`` / ``GpuFusedAdamAdapter`` accept a single
    flat param list + a single ``weight_decay`` scalar, so the underlying
    ``torch.optim.Optimizer`` ends up with exactly one param group whose
    ``weight_decay`` applies uniformly to every param. To preserve the
    HF Trainer.create_optimizer convention (bias/LayerNorm in a
    ``weight_decay=0.0`` group), we post-process each underlying
    optimizer's ``param_groups`` here: for any group containing at least
    one no-decay param AND at least one decay param, we split it into
    two groups — same hyperparams except the no-decay group's
    ``weight_decay`` is forced to ``0.0``. Single-membership groups
    (all-decay or all-no-decay) get their ``weight_decay`` set in place
    without an extra group.

    No-op when ``inner`` is ``None`` (empty-param adapter), when the
    no-decay set is empty, or when no group needs splitting.
    """
    if inner is None or not no_decay_param_ids:
        return
    new_groups: list[dict[str, Any]] = []
    changed = False
    for group in inner.param_groups:
        params = list(group["params"])
        decay_params = [p for p in params if id(p) not in no_decay_param_ids]
        no_decay_params = [p for p in params if id(p) in no_decay_param_ids]
        if not no_decay_params:
            # Fully-decay group: leave weight_decay as the caller set it.
            new_groups.append(group)
            continue
        if not decay_params:
            # Fully-no-decay group: zero its weight_decay in place.
            if group.get("weight_decay", 0.0) != 0.0:
                group["weight_decay"] = 0.0
                changed = True
            new_groups.append(group)
            continue
        # Mixed: split into two groups sharing every other hyperparam.
        decay_group = {**group, "params": decay_params}
        no_decay_group = {**group, "params": no_decay_params, "weight_decay": 0.0}
        new_groups.append(decay_group)
        new_groups.append(no_decay_group)
        changed = True
    if not changed:
        return
    # ``torch.optim.Optimizer`` stores param_groups as a list of dicts and
    # ``step()`` reads ``group["weight_decay"]`` per group, so direct
    # replacement is safe. Per-param state lives in ``optimizer.state``
    # keyed by parameter ``id``, not by group index, so re-grouping the
    # same params across two groups doesn't disturb existing moment
    # buckets (we run this before the first step anyway — adapters are
    # freshly built above and have no state yet).
    inner.param_groups = new_groups


#: Axolotl / HF Trainer optimizer-name strings that route the persistent
#: chunk set through ``GpuAdamW8bitAdapter`` instead of
#: ``GpuFusedAdamAdapter``. ``adamw_8bit`` and ``adamw_bnb_8bit`` are
#: aliases in HF's ``OptimizerNames`` (training_args.py:128-129) that both
#: dispatch to ``bnb.optim.AdamW`` with ``optim_bits=8``; we accept both
#: spellings so users carrying configs from either origin work without
#: edits. ``paged_adamw_8bit`` selects the paged variant (UVM-backed
#: state) for the same persistent set.
_BNB_8BIT_OPTIMIZERS: frozenset[str] = frozenset(
    {"adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit"}
)
_BNB_8BIT_PAGED_OPTIMIZERS: frozenset[str] = frozenset({"paged_adamw_8bit"})


def _normalize_optimizer_name(name: str | None) -> str | None:
    """Lower-case + strip whitespace; ``None`` passes through unchanged.

    Centralised so both the public dispatch check below and any future
    callers (e.g. checkpoint resume) compare against the same normalised
    representation.
    """
    if name is None:
        return None
    return str(name).strip().lower()


def protrain_optimizer_wrapper(
    wrapped: WrappedModel,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    optimizer_name: str | None = None,
) -> torch.optim.Optimizer:
    """Rebuild the GPU/CPU FusedAdam adapters at user-specified hyperparams.

    ``protrain_model_wrapper`` instantiates transient adapters with
    placeholder hyperparams so the chunk manager has something to drive
    during bring-up. This function rebuilds them with the real
    ``lr`` / ``betas`` / ``eps`` / ``weight_decay``, then swaps them
    into the chunk manager in-place so the scheduler's async
    ``reduce_grads_and_offload`` path continues to pump the right
    optimizer.

    The HF Trainer's ``create_optimizer`` splits parameters into a
    decay group and a ``weight_decay=0.0`` group for bias / LayerNorm /
    RMSNorm params. We honor that split here by post-processing each
    underlying torch ``Optimizer.param_groups`` after adapter
    construction (see :func:`_split_optim_param_groups`); the supplied
    ``weight_decay`` argument applies only to the decay group.

    Sharded path (``zero3_shard=True``): the CPU adapter steps over each
    chunk's per-region flat ``shard_param`` rather than the original
    ``nn.Parameter`` objects, so a direct id-match against the
    no-decay source set finds nothing. We bridge that gap in
    :func:`_collect_sharded_no_decay_shard_param_ids` by walking
    ``ChunkManager._cpu_slots`` (which carries ``param_id`` +
    ``byte_offset`` + ``numel * element_size`` per param) and
    intersecting each slot's byte range against each region's
    ``[chunk_offset, chunk_offset + region_bytes)`` extent: any region
    overlapping at least one no-decay source param has its
    ``shard_param`` added to the no-decay set fed to
    :func:`_split_optim_param_groups`. This is correctness-conservative
    — we may carry a few wd=decay bytes inside a region pinned to wd=0
    by an adjoining norm scale, but we never silently decay a bias or
    norm param the upstream Trainer would have left at ``wd=0``
    (CR PR #17 R3190973417).
    """
    chunk_manager = cast("ChunkManager", wrapped.chunk_manager)
    layout = chunk_manager.layout
    persistent_ids = set(chunk_manager._persistent_ids)

    # Partition params the same way ``protrain_model_wrapper`` did —
    # persistent chunks go to GPU FusedAdam, the rest to per-chunk
    # CPU FusedAdam adapters. Membership-test against the chunk
    # manager's actual ``_persistent_ids`` set rather than a prefix
    # ``cid < n_persist`` test: non-block-chunk pinning expands the
    # persistent set into a non-contiguous shape (e.g. {0..110, 129}
    # when an untied lm_head lands at chunk 129), and a prefix test
    # would mis-route the high-cid persistent chunk's GPU params to
    # CPU FusedAdam — which materialize_offload never offloaded, so
    # the CPU adam would step against full-size GPU tensors and the
    # mid-prefix non-persistent chunk's CPU shards would never get
    # an optimizer step.
    # Resolve params via ChunkManager._params_by_id (populated at chunk-
    # manager construction, which runs PRE-block-wrap) rather than
    # ``module.named_parameters()`` (which after wrapping carries a
    # ``.block.`` infix from the OffloadedBlock/SwappedBlock/CheckpointedBlock
    # wrappers, mismatching the layout's pre-wrap pid keys). Without this
    # fix, the per-chunk param list comes back empty for any wrapped
    # block — silently skipping optimizer construction for those chunks
    # and leading to ``cpu_optim is None`` at backward (R2-05 fail-fast).
    persistent_params: list["nn.Parameter"] = []
    cpu_params_per_chunk: dict[ChunkId, list["nn.Parameter"]] = {}

    for cid, chunk_param_ids in enumerate(layout.chunks):
        # Fail fast on unresolvable pids: silently dropping them produced
        # partial freezing (a chunk's optimizer would only step the params
        # that happened to be registered) and wedged debugging because the
        # symptom appeared far downstream as "no gradient update on
        # weight X". Raise here so the user sees the exact chunk and pid.
        missing_ids = [
            pid for pid in chunk_param_ids if pid not in chunk_manager._params_by_id
        ]
        if missing_ids:
            raise ValueError(
                f"chunk cid={cid} references param ids {missing_ids} that are "
                "not registered in ChunkManager._params_by_id; cannot build "
                "per-chunk optimizer (would silently skip these params). "
                "Known pids: "
                f"{sorted(chunk_manager._params_by_id.keys())[:8]}"
                f"{'...' if len(chunk_manager._params_by_id) > 8 else ''}"
            )
        chunk_params = [chunk_manager._params_by_id[pid] for pid in chunk_param_ids]
        if cid in persistent_ids:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[ChunkId(cid)] = chunk_params

    # M2.5 dispatch — pair 8-bit weight quantization with 8-bit optimizer
    # state when the user requested an Axolotl/HF ``adamw_8bit`` /
    # ``adamw_bnb_8bit`` / ``paged_adamw_8bit`` optimizer name. Bail
    # condition: bnb 8-bit Adam kernels run on CUDA only, so only the
    # persistent (GPU-resident) chunk set can use the 8-bit adapter; the
    # non-persistent CPU shards keep the existing 32-bit DeepSpeedCPUAdam
    # path and we surface a one-shot warning so users see the partial
    # win (phase2.md §M2.5).
    normalized_optim_name = _normalize_optimizer_name(optimizer_name)
    use_bnb_8bit = normalized_optim_name in _BNB_8BIT_OPTIMIZERS
    use_paged_8bit = normalized_optim_name in _BNB_8BIT_PAGED_OPTIMIZERS

    gpu_optim: GpuFusedAdamAdapter | GpuAdamW8bitAdapter | None = None
    cpu_optim: CpuFusedAdamAdapter | None = None
    if persistent_params:
        if use_bnb_8bit:
            LOG.info(
                "protrain_optimizer_wrapper: routing %d persistent params "
                "through bnb %s (optimizer_name=%s)",
                len(persistent_params),
                "PagedAdamW8bit" if use_paged_8bit else "AdamW8bit",
                optimizer_name,
            )
            gpu_optim = GpuAdamW8bitAdapter(
                params=persistent_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                paged=use_paged_8bit,
            )
        else:
            gpu_optim = GpuFusedAdamAdapter(
                params=persistent_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )

    # M7: for sharded non-persistent chunks the CPU Adam updates each
    # :class:`_DtypeRegion`'s flat shard_param (one per rank slice per
    # dtype region) rather than the user-facing per-param list.
    # Homogeneous-dtype chunks have exactly one region and behave
    # identically to the pre-followup path; mixed-dtype chunks expose
    # one shard_param per region.
    cpu_params_per_chunk_for_optim: dict[ChunkId, list["nn.Parameter"]] = {}
    for cid, chunk_params in cpu_params_per_chunk.items():
        shard_state = chunk_manager._chunk_shards.get(cid)
        if shard_state is not None and shard_state.regions:
            cpu_params_per_chunk_for_optim[cid] = [
                r.shard_param for r in shard_state.regions
            ]
        else:
            cpu_params_per_chunk_for_optim[cid] = chunk_params

    if use_bnb_8bit and any(
        params for params in cpu_params_per_chunk_for_optim.values()
    ):
        # Bail criterion (phase2.md §M2.5): bnb 8-bit Adam requires CUDA
        # tensors; non-persistent chunks live on CPU. We keep the
        # 32-bit CpuFusedAdamAdapter on those chunks so training stays
        # correct (and the user still gets the persistent-chunk 8-bit
        # win from above). Surface this once, loudly, so users
        # configuring `adamw_8bit` aren't surprised by the partial
        # adoption.
        n_cpu_chunks = sum(
            1 for params in cpu_params_per_chunk_for_optim.values() if params
        )
        LOG.warning(
            "protrain_optimizer_wrapper: optimizer_name=%s requested 8-bit "
            "AdamW, but %d non-persistent chunk(s) live on CPU and bnb's "
            "8-bit Adam kernels are CUDA-only. Those chunks will keep "
            "using 32-bit DeepSpeedCPUAdam (still correct, but the "
            "optimizer-state memory win applies only to the persistent "
            "set). To get end-to-end 8-bit, configure ProTrain with all "
            "chunks persistent (Mode A) — e.g. set "
            "protrain_force_all_persistent: true.",
            optimizer_name,
            n_cpu_chunks,
        )

    if any(params for params in cpu_params_per_chunk_for_optim.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk_for_optim,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        except Exception as err:
            # Only ``ImportError`` (DeepSpeed not installed) and
            # ``CUDAMismatchException`` (a subclass of ``Exception``, not
            # ``ImportError``, raised when system CUDA disagrees with
            # torch's CUDA wheel) get translated into the install-DeepSpeed
            # error path; any other exception is a real bug in
            # ``CpuFusedAdamAdapter`` initialization and must propagate
            # unchanged so it is not silently masked. We compare the
            # CUDAMismatch class name as a string to avoid a hard import
            # on a broken deepspeed install.
            is_cuda_mismatch = type(err).__name__ == "CUDAMismatchException"
            if not isinstance(err, ImportError) and not is_cuda_mismatch:
                raise
            # Render the exception to a string before logging — passing
            # the live ``err`` object into LOG.error propagates
            # ``err.__traceback__`` → frame locals (the persistent /
            # cpu-resident param lists in this scope) into LogRecord.args.
            # Test runners that retain log records would then leak one
            # full model footprint per failed wrap. The ``raise ... from
            # err`` below is fine — that hands ``err`` to the caller's
            # except path, not the logger's record retention.
            err_kind = type(err).__name__
            err_str = str(err)
            base_msg = (
                "protrain_optimizer_wrapper: CPU FusedAdam unavailable "
                "(%s: %s). Non-persistent chunks will NOT receive "
                "optimizer steps — only persistent chunks (the GPU "
                "optimizer) update. Training is incorrect in this "
                "state for any model whose non-persistent params "
                "matter for convergence."
            )
            if is_cuda_mismatch:
                LOG.error(
                    base_msg + " Detected DeepSpeed CUDAMismatchException — "
                    "system CUDA does not match torch's CUDA wheel. "
                    "Workaround: set env DS_SKIP_CUDA_CHECK=1 (CPU Adam "
                    "JIT-compiles correctly despite the mismatch on "
                    "most rigs).",
                    err_kind,
                    err_str,
                )
            else:
                LOG.error(
                    base_msg + " Install DeepSpeed (or fix its dependencies) to "
                    "enable async CPU Adam.",
                    err_kind,
                    err_str,
                )
            raise RuntimeError(
                "CpuFusedAdamAdapter is required whenever ProTrain has "
                "non-persistent chunks (cpu_params_per_chunk_for_optim "
                "is non-empty); without it those offloaded params receive "
                "computed gradients but never an optimizer step, silently "
                "corrupting training. Fix the DeepSpeed install (e.g., set "
                "DS_SKIP_CUDA_CHECK=1 if this is a CUDA-toolkit / "
                "torch-wheel mismatch) or switch to an all-persistent "
                "config so no CPU optimizer is needed."
            ) from err

    # Preserve HF Trainer's bias/norm no-decay split — the adapter
    # constructors take a single ``weight_decay`` scalar, so we
    # post-process each underlying torch Optimizer's param_groups to
    # split out the no-decay subset. ``model_wrapper.py`` resolves
    # ``wrapped.module`` to the original (pre-block-wrap) ``nn.Module``,
    # which is the same names ``named_parameters()`` returned at chunk
    # build time, so id-membership matches the GPU optim's persistent
    # params directly. For the CPU optim's sharded chunks the shard_param
    # ids do NOT match the original-param ids, so we bridge with
    # :func:`_collect_sharded_no_decay_shard_param_ids` (region byte
    # intersection); see its docstring for the correctness argument.
    no_decay_param_ids = _collect_no_decay_param_ids(wrapped.module)
    if no_decay_param_ids:
        if gpu_optim is not None:
            _split_optim_param_groups(gpu_optim.underlying, no_decay_param_ids)
        if cpu_optim is not None:
            sharded_no_decay_ids = _collect_sharded_no_decay_shard_param_ids(
                chunk_manager,
                cpu_params_per_chunk,
                no_decay_param_ids,
            )
            # Union: original-param ids cover the homogeneous-replicated
            # path (where the CPU adapter holds the original nn.Parameters),
            # shard_param ids cover the M7 sharded path. A given inner
            # optimizer only sees one set or the other, so the union is
            # always disjoint at lookup time.
            cpu_no_decay_ids = no_decay_param_ids | sharded_no_decay_ids
            # ``CpuFusedAdamAdapter`` exposes per-chunk inner optimizers via
            # the (private) ``_optims`` dict; there's no public iterator,
            # and adding one would touch a sibling file. ``getattr`` keeps
            # this resilient if a future refactor renames the slot.
            inner_optims = getattr(cpu_optim, "_optims", {}) or {}
            for inner in inner_optims.values():
                _split_optim_param_groups(inner, cpu_no_decay_ids)

    # Swap the freshly-built adapters into the chunk manager so the
    # scheduler's post_block_backward -> reduce_grads_and_offload ->
    # cpu_optim.step_async chain uses them. The chunk manager's
    # ``gpu_optim`` slot is typed ``GpuFusedAdamAdapter | None`` (the
    # legacy adapter); the M2.5 ``GpuAdamW8bitAdapter`` is duck-compat
    # at the call sites that consume the slot (``.step()``,
    # ``.zero_grad()``, ``.state_dict()`` — see
    # :class:`GpuAdamW8bitAdapter`). We assign through a typing cast
    # rather than widening the chunk manager's type signature, which
    # would touch a read-only file from this milestone's perspective.
    chunk_manager.cpu_optim = cpu_optim
    chunk_manager.gpu_optim = cast("GpuFusedAdamAdapter | None", gpu_optim)

    # Build the flat param list for the Optimizer base class.
    all_params: list["nn.Parameter"] = list(persistent_params)
    for params in cpu_params_per_chunk.values():
        all_params.extend(params)
    # Dedupe while preserving order — shared weights may appear twice.
    seen: set[int] = set()
    unique_params: list["nn.Parameter"] = []
    for p in all_params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        unique_params.append(p)

    defaults: dict[str, Any] = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return _ProTrainOptimizer(
        gpu_optim=gpu_optim,
        cpu_optim=cpu_optim,
        params=unique_params,
        defaults=defaults,
        chunk_manager=chunk_manager,
    )


__all__ = ["protrain_optimizer_wrapper"]
