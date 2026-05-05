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
* ``state_dict`` / ``load_state_dict`` — explicitly raise
  ``NotImplementedError``. Optimizer-state checkpointing is M5/M6
  scope; the M4b contract is to keep the method names resolvable so
  HuggingFace Trainer does not blow up if it touches the optimizer
  during init.
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

    # ---- checkpointing: deliberately unimplemented for M4 ---------------

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Reject the call — checkpointing goes through the dedicated callback (M5/M6)."""
        raise NotImplementedError(
            "ProTrain optimizer checkpointing is M5/M6 work; "
            "disable optimizer-state saving for now."
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Reject the call — checkpointing goes through the dedicated load hook (M5/M6)."""
        raise NotImplementedError(
            "ProTrain optimizer checkpointing is M5/M6 work; "
            "disable optimizer-state loading for now."
        )


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
        except (ImportError, Exception) as err:  # noqa: BLE001 - see below
            # DeepSpeed's CUDA-version mismatch raises a
            # ``CUDAMismatchException`` (subclass of ``Exception``, not
            # ``ImportError``). Compare by class name to avoid a hard
            # import on a broken deepspeed install.
            is_cuda_mismatch = type(err).__name__ == "CUDAMismatchException"
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
