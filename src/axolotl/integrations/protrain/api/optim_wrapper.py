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
* ``state_dict`` / ``load_state_dict`` — torch-side no-ops. The
  adapters own their own state and persist it through the dedicated
  ProTrain checkpoint hook (M5/M6); ``state_dict`` returns the empty
  ``{"state": {}, "param_groups": [...]}`` shell HF Trainer +
  Accelerate expect at ``prepare`` time, and ``load_state_dict``
  accepts and silently discards the round-tripped payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from axolotl.integrations.protrain.chunk import (
    CpuFusedAdamAdapter,
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
        gpu_optim: GpuFusedAdamAdapter | None,
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

    def step(self, closure: Any = None) -> Any:  # noqa: ARG002 — HF convention
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
        """
        # Orphan sweep: ensure every non-persistent chunk has been
        # reduced+offloaded before we wait. See the docstring above for
        # why this is necessary in the sharded path.
        cm = self._chunk_manager
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

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return an empty torch-side optimizer state.

        Real ProTrain optimizer state (per-shard moments held inside the
        CPU/GPU FusedAdam adapters) is saved by the dedicated checkpoint
        callback, not through this method. We still preserve HF's
        ``{"state": ..., "param_groups": ...}`` shape so Accelerate's
        ``move_to_device(state_dict, ...)`` + ``load_state_dict`` round
        trip at ``prepare`` time does not crash.
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
        return {"state": {}, "param_groups": param_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Accept and discard torch-side state.

        The dedicated ProTrain load hook restores adapter state from the
        checkpoint shard files; the torch-facing ``state_dict`` we just
        returned is empty by construction, so silently dropping the
        round-tripped payload is correct.
        """
        return None


# HF Trainer's ``get_decay_parameter_names`` excludes bias and norm-layer
# parameters from weight decay by default; if we collapse everything into
# a single global ``weight_decay`` here we silently change training behavior
# relative to the stock Trainer path. The token list below mirrors HF's
# name-based filter (``bias``, ``LayerNorm``, ``RMSNorm``, ``.norm.``,
# ``_norm``) and is matched case-insensitively against
# ``model.named_parameters()`` names.
_HF_NO_DECAY_NAME_TOKENS: tuple[str, ...] = (
    "bias",
    "layernorm",
    "rmsnorm",
    ".norm.",
    "_norm",
)


def _collect_no_decay_param_ids(module: "nn.Module") -> set[int]:
    """Return ``id(p)`` for every parameter HF Trainer would put in the no-decay group.

    Mirrors :func:`transformers.trainer_pt_utils.get_decay_parameter_names`
    by filtering parameter NAMES against
    ``_HF_NO_DECAY_NAME_TOKENS``. Name-based matching (case-insensitive)
    catches both LayerNorm/RMSNorm modules and bias terms — the same set
    that the upstream Trainer puts in its ``weight_decay=0.0`` group.
    """
    no_decay: set[int] = set()
    for name, param in module.named_parameters():
        lname = name.lower()
        if any(tok in lname for tok in _HF_NO_DECAY_NAME_TOKENS):
            no_decay.add(id(param))
    return no_decay


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


def protrain_optimizer_wrapper(
    wrapped: WrappedModel,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
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

    Caveat — sharded path: when ``zero3_shard=True`` the CPU adapter is
    built against each chunk's flat per-region ``shard_param`` rather
    than the original ``nn.Parameter`` objects, so we cannot identify
    bias/norm BYTES inside a shard_param post-hoc. ``_DtypeRegion``
    splitting on dtype already isolates fp32 norm regions from fp16
    attention/MLP regions on the standard Llama config, but bias terms
    that share their parent linear's dtype remain in the decay group in
    that mode. Splitting regions on decay-membership requires touching
    ``ChunkManager.materialize_offload`` and is deferred.
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
        chunk_params = [
            chunk_manager._params_by_id[pid]
            for pid in chunk_param_ids
            if pid in chunk_manager._params_by_id
        ]
        if cid in persistent_ids:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[ChunkId(cid)] = chunk_params

    gpu_optim: GpuFusedAdamAdapter | None = None
    cpu_optim: CpuFusedAdamAdapter | None = None
    if persistent_params:
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
    # build time, so id-membership matches what the adapters now hold.
    no_decay_param_ids = _collect_no_decay_param_ids(wrapped.module)
    if no_decay_param_ids:
        if gpu_optim is not None:
            _split_optim_param_groups(gpu_optim.underlying, no_decay_param_ids)
        if cpu_optim is not None:
            # ``CpuFusedAdamAdapter`` exposes per-chunk inner optimizers via
            # the (private) ``_optims`` dict; there's no public iterator,
            # and adding one would touch a sibling file. ``getattr`` keeps
            # this resilient if a future refactor renames the slot.
            inner_optims = getattr(cpu_optim, "_optims", {}) or {}
            for inner in inner_optims.values():
                _split_optim_param_groups(inner, no_decay_param_ids)

    # Swap the freshly-built adapters into the chunk manager so the
    # scheduler's post_block_backward -> reduce_grads_and_offload ->
    # cpu_optim.step_async chain uses them.
    chunk_manager.cpu_optim = cpu_optim
    chunk_manager.gpu_optim = gpu_optim

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
