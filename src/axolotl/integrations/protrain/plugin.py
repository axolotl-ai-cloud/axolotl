# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BasePlugin subclass for ProTrain (M5, DESIGN.md §Plugin Integration).

Thin shim over the M1-M4 runtime primitives: wires Axolotl's plugin hook
points (``post_model_load`` / ``create_optimizer`` / ``post_trainer_create``)
to ``protrain_model_wrapper`` / ``protrain_optimizer_wrapper``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.protrain.api.hardware import (  # noqa: F401 — re-exported for back-compat
    DEFAULT_PCIE_BPS as _DEFAULT_PCIE_BPS,
    build_hardware_profile as _shared_build_hardware_profile,
    resolve_world_size_from_env as _resolve_world_size_from_env,
)
from axolotl.integrations.protrain.args import _has_protrain_plugin
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from transformers import Trainer

    from axolotl.integrations.protrain.chunk import ChunkManager

LOG = get_logger(__name__)


# ``_DEFAULT_PCIE_BPS`` and ``_resolve_world_size_from_env`` are re-exported
# from :mod:`axolotl.integrations.protrain.api.hardware` so the plugin path
# and the direct API helper (:func:`auto_wrap`) share a single canonical
# source. The leading-underscore aliases preserve the legacy import paths
# any external caller (or future drift detector) may have keyed on.


def _early_init_dist_for_nccl(cfg) -> int:
    """Initialise ``torch.distributed`` from env-derived rendezvous if needed.

    Item 6 — Preflight NCCL measurement. The paper's cost model takes
    real per-payload NCCL gather/reduce times as load-bearing inputs to
    the search; running the searcher with empty tables drives a wrong
    Mode-C config on multi-rank workloads. The fix: bring the process
    group up *before* :func:`protrain_model_wrapper` so the trace's call
    to :func:`profiler.hw_bench.measure_nccl` records real timings on
    the live PG.

    Skip rules:

    * ``WORLD_SIZE <= 1`` — single-rank, no NCCL traffic. Returns 1.
    * ``LOCAL_RANK`` / ``RANK`` unset — we are not under torchrun /
      Accelerate's launcher, so the rendezvous env we'd need (``MASTER_ADDR``,
      ``MASTER_PORT``) is missing. Returns 1.
    * ``cfg.ddp_backend`` set to a non-default backend — the user has
      asked for a specific backend; an early ``"nccl"`` init would lock
      them out. Defer to Accelerate / HF Trainer. Returns 1.
    * CUDA unavailable — NCCL needs GPU tensors. Returns 1.
    * ``torch.distributed.is_initialized()`` already True — somebody
      else (Accelerate's prior call from a previous test, a custom
      launcher, …) brought the PG up. Returns the live world size.

    Otherwise calls ``dist.init_process_group(backend="nccl")`` against
    the env-derived rendezvous and returns the world size. Accelerate's
    later ``Accelerator()`` constructor checks ``is_initialized()`` and
    skips its own init when we've already brought the PG up — see
    ``accelerate/state.py`` ``PartialState.__init__`` lines 219-244.

    Returns
    -------
    int
        The effective world size (1 means "treat as single-rank, do not
        run NCCL premeasure").
    """
    import os

    world_size = _resolve_world_size_from_env()
    if world_size <= 1:
        return 1

    # Sanity-check the launcher provided enough env to rendezvous. A
    # bare ``WORLD_SIZE > 1`` without ``LOCAL_RANK`` typically indicates
    # a misconfigured manual export rather than a real torchrun-managed
    # process; bail rather than crash inside ``init_process_group``.
    if os.environ.get("LOCAL_RANK") is None or os.environ.get("RANK") is None:
        LOG.warning(
            "ProTrain: WORLD_SIZE=%d but LOCAL_RANK/RANK not set — assuming "
            "non-launcher environment, skipping early dist init. NCCL "
            "tables will be empty and Mode-C selection may be suboptimal.",
            world_size,
        )
        return 1

    # Custom backend opt-out. ``cfg.ddp_backend`` mirrors HF
    # ``TrainingArguments.ddp_backend`` (passed through Axolotl's
    # ``training_args.py``); when the user has specified a non-default
    # backend, they explicitly want Accelerate / HF to own the init
    # call, and our early ``"nccl"`` init would clobber it.
    ddp_backend = getattr(cfg, "ddp_backend", None)
    if ddp_backend not in (None, "", "nccl"):
        LOG.info(
            "ProTrain: cfg.ddp_backend=%r is non-default; skipping early "
            "dist init. The deferred late-bind path "
            "(_remeasure_nccl_and_research) will splice NCCL tables once "
            "the trainer brings the PG up.",
            ddp_backend,
        )
        return 1

    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return 1

    if not dist.is_available():
        LOG.warning(
            "ProTrain: torch.distributed unavailable but WORLD_SIZE=%d. "
            "Skipping early dist init.",
            world_size,
        )
        return 1

    if dist.is_initialized():
        # Some other path (Accelerate from a prior cfg, a custom
        # launcher) already brought the PG up. Skip our init but do
        # surface the live world for downstream callers.
        try:
            return int(dist.get_world_size())
        except (RuntimeError, ValueError):
            return world_size

    if not torch.cuda.is_available():
        # NCCL backend requires CUDA; if we lack it, skip the init and
        # let the late-bind path (or a Gloo-based test harness) handle
        # it.
        LOG.info(
            "ProTrain: CUDA unavailable; skipping early NCCL dist init "
            "(WORLD_SIZE=%d).",
            world_size,
        )
        return 1

    # Bind this rank to its local GPU before initialising NCCL so the
    # default device used for collectives matches the per-rank shard. HF
    # Trainer / Accelerate normally do this themselves later, but our
    # early ``measure_nccl`` (called by ``run_trace``) issues GPU-side
    # collectives and must see the correct device on entry. ``LOCAL_RANK``
    # is the per-host ordinal under torchrun; under
    # ``CUDA_VISIBLE_DEVICES`` it indexes into the masked subset.
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    except (ValueError, RuntimeError) as exc:
        LOG.warning(
            "ProTrain: torch.cuda.set_device(LOCAL_RANK=%s) failed (%s); "
            "early dist init may pick the wrong device.",
            os.environ.get("LOCAL_RANK"),
            exc,
        )

    LOG.info(
        "ProTrain: bringing up torch.distributed (backend=nccl, "
        "world_size=%d, rank=%s, local_rank=%s) ahead of the wrapper so "
        "the profiler trace captures real NCCL gather/reduce times "
        "(paper §3.3 / Appendix A). Accelerate's later Accelerator() "
        "will detect is_initialized()=True and skip re-initialising.",
        world_size,
        os.environ.get("RANK"),
        os.environ.get("LOCAL_RANK"),
    )
    try:
        dist.init_process_group(backend="nccl")
    except (RuntimeError, ValueError) as exc:
        LOG.warning(
            "ProTrain: early dist.init_process_group(backend=nccl) failed "
            "(%s); falling back to the late-bind NCCL re-measurement path.",
            exc,
        )
        return 1

    try:
        live_world = int(dist.get_world_size())
    except (RuntimeError, ValueError):
        live_world = world_size
    return live_world


def _remeasure_nccl_and_research(wrapped) -> tuple[bool, bool]:
    """Late-bind real NCCL timings into the cached trace, then re-run search().

    **Role under Item 6 (post-2026-04 preflight flow):** defensive
    fallback. The primary path now lives in
    :func:`_early_init_dist_for_nccl` + :func:`post_model_load`: the
    plugin brings the process group up *before* invoking the wrapper,
    so the trace's call to :func:`profiler.hw_bench.measure_nccl`
    captures real NCCL times on the live PG and the search picks the
    correct config from the start. This helper still runs from
    ``post_trainer_create`` to handle the cases where early init was
    skipped — non-default ``cfg.ddp_backend``, user-supplied process
    group, CPU-only test runs that bring up Gloo later, etc. — so the
    cost model is never left consuming empty tables on a real
    multi-rank workload. With the early-init path active, this branch
    is normally a no-op (the trace's NCCL tables are populated and the
    idempotency check below short-circuits).

    The legacy commentary, retained for context: previously the default
    Axolotl plugin path ran ``protrain_model_wrapper`` from
    ``post_model_load`` *before* dist init, so the profiler short-circuited
    to empty tables and the trace recorded ``world=1`` regardless of the
    eventual world size. Mode C (ZeRO-3 sharded) consumes the NCCL tables
    in ``cost/runtime.estimate_runtime``; with empty tables, sharded
    predictions under-counted the per-chunk gather + reduce-scatter cost.

    On invocation, the helper measures NCCL on the live process group,
    splices the new tables and actual world size into the cached trace,
    persists the updated trace under a new cache key, and re-runs
    ``search()`` with the same layout + capacity + hardware profile.
    Behaviour after the re-run depends on whether the picked config
    actually moved:

    * **Same cfg + block_map (the expected case post-Item 6).** Only
      the predicted iter time and the trace's NCCL tables refreshed,
      so it is safe to publish them onto ``WrappedModel.search_result``
      / ``_trace`` — the installed runtime still matches.
    * **Different cfg or block_map.** The chunk_manager / scheduler /
      hooks (and the optimizer state slots that ride on them) are
      already wired for the bootstrap config; rebuilding mid-flight
      would invalidate them. Instead of overwriting the live runtime
      contract, the late-search outputs are stashed on
      ``post_nccl_search_result`` / ``post_nccl_trace`` (telemetry
      only) and a DEBUG (was WARNING pre-Item 6) is logged. The
      installed ``search_result`` / ``_trace`` continue to reflect
      what is actually running. Future runs hit the multi-rank cache
      and pick the new config from the start.

    Returns ``(updated, cfg_changed)`` for telemetry / test inspection:

    * ``updated`` — True iff the trace's NCCL tables were rewritten
      (False on single-rank, on missing dist init, or when the trace
      already had populated tables).
    * ``cfg_changed`` — True iff the re-run search picked a different
      ``cfg`` or ``block_map`` than the original. Implies ``updated``.
    """
    import dataclasses

    try:
        import torch.distributed as dist
    except ImportError:
        return (False, False)

    if not dist.is_available() or not dist.is_initialized():
        return (False, False)
    world_size = int(dist.get_world_size())
    if world_size <= 1:
        return (False, False)

    trace = getattr(wrapped, "_trace", None)
    layout = getattr(wrapped, "_layout", None)
    hw = getattr(wrapped, "_hardware_profile", None)
    capacity = getattr(wrapped, "_capacity_bytes", None)
    cache_key = getattr(wrapped, "_cache_key", None)
    if (
        trace is None
        or layout is None
        or hw is None
        or capacity is None
        or cache_key is None
    ):
        LOG.warning(
            "ProTrain: NCCL re-measurement skipped — wrapped model is "
            "missing one of {_trace,_layout,_hardware_profile,"
            "_capacity_bytes,_cache_key}. Cost-model NCCL terms will fall back to "
            "the empty-table path."
        )
        return (False, False)

    # Idempotency: if the cached trace already carries NCCL tables (e.g.
    # second call on a re-entrant trainer create, or a cache hit on a
    # prior multi-rank run), skip the measurement but DO consider the
    # re-run search a no-op.
    if trace.nccl_gather_s and trace.nccl_reduce_s and trace.world == world_size:
        return (False, False)

    # With overrides pinning the plan, late NCCL re-search would raise on a cost-optimal cfg that differs from the bootstrap.
    if bool(getattr(wrapped, "_override_skip_trace", False)):
        LOG.info(
            "ProTrain: late NCCL re-search skipped — explicit override knobs "
            "are fully set so the bootstrap cfg is pinned. world_size=%d, "
            "bootstrap cfg=%s.",
            world_size,
            wrapped.search_result.cfg,
        )
        return (False, False)

    from axolotl.integrations.protrain.profiler import measure_nccl
    from axolotl.integrations.protrain.profiler.cache import (
        ProfilerCacheKey,
        save_cached_trace,
    )
    from axolotl.integrations.protrain.search import search

    LOG.info(
        "ProTrain: re-measuring NCCL on world_size=%d (trace was profiled "
        "with empty tables)",
        world_size,
    )
    try:
        gather_table, reduce_table = measure_nccl(world_size)
    except (RuntimeError, ImportError) as exc:
        LOG.warning(
            "ProTrain: NCCL re-measurement failed (%s); leaving trace "
            "with empty tables — Mode C predictions will under-count "
            "comm cost.",
            exc,
        )
        return (False, False)

    new_trace = dataclasses.replace(
        trace,
        nccl_gather_s=gather_table,
        nccl_reduce_s=reduce_table,
        world=world_size,
    )

    # Save under a new cache key with the live world so future multi-
    # rank runs skip the round-trip. Leave the original world=1 entry
    # alone (it is the correct cache for single-rank runs).
    new_key = ProfilerCacheKey(
        arch_hash=cache_key.arch_hash,
        bs=cache_key.bs,
        seq=cache_key.seq,
        sku=cache_key.sku,
        world=world_size,
    )
    try:
        save_cached_trace(
            new_key,
            new_trace,
            cache_dir=getattr(wrapped, "_cache_dir", None),
        )
    except OSError as exc:
        LOG.warning(
            "ProTrain: failed to persist updated trace to cache (%s); "
            "the in-memory trace is still updated for this run.",
            exc,
        )

    # Re-run search with the populated tables. ``hw`` is reused as-is —
    # gpu_count was already correct at wrapper time (hw.gpu_count was
    # set from torch.cuda.device_count(), which under torchrun is the
    # per-rank device count, not the world size; the searcher reads
    # ``trace.world`` for the comm-cost gate). Reuse the same per-rank
    # CPU feasibility budget the original search consumed; ``None``
    # means the wrapper deferred to the GPU-only filter (e.g. psutil
    # missing) and the re-search should mirror that.
    cpu_capacity = getattr(wrapped, "_cpu_capacity_bytes", None)
    new_result = search(
        new_trace, layout, capacity, hw, cpu_capacity_bytes=cpu_capacity
    )

    cfg_changed = (
        new_result.cfg != wrapped.search_result.cfg
        or new_result.block_map != wrapped.search_result.block_map
    )
    if cfg_changed:
        # With Item 6's preflight NCCL measurement (early
        # ``dist.init_process_group`` in ``post_model_load``), the late
        # re-search should normally be a no-op: the trace already
        # carries real NCCL tables and the search runs on accurate cost
        # inputs. Hitting this branch means the accurate NCCL search
        # picked a different plan than the bootstrap, but the
        # chunk_manager / scheduler / hooks (and the optimizer state
        # slots that ride on them) are already wired for the bootstrap
        # config and cannot be rebuilt mid-flight. Continuing under the
        # bootstrap plan would silently train under a config the
        # accurate search no longer endorses (CR PR #19); fail-fast
        # instead so the user fixes the early-dist-init path. Telemetry
        # is still stashed so tests / post-mortem inspection can read
        # both plans off the WrappedModel before the exception unwinds.
        LOG.warning(
            "ProTrain: post-NCCL search picked a different config than "
            "the bootstrap prediction. cfg %s -> %s; stashing the "
            "post-NCCL plan on WrappedModel.post_nccl_search_result for "
            "telemetry. Reaching this branch suggests early dist init "
            "was skipped — check cfg.ddp_backend / launcher env.",
            wrapped.search_result.cfg,
            new_result.cfg,
        )
        # Telemetry-only: keep the late-search outputs visible to
        # callers (tests, dynamic re-tuning) BEFORE we raise, so the
        # raised exception's caller can introspect both plans.
        wrapped.post_nccl_search_result = new_result  # type: ignore[attr-defined]
        wrapped.post_nccl_trace = new_trace  # type: ignore[attr-defined]
        raise RuntimeError(
            "ProTrain: late NCCL re-search picked a different plan than "
            "the bootstrap. Continuing would silently train under a "
            "config the accurate search no longer endorses (the "
            "chunk_manager / scheduler / hooks / optimizer state slots "
            "are already wired for the bootstrap plan and cannot be "
            "rebuilt mid-flight).\n"
            f"  bootstrap cfg: {wrapped.search_result.cfg}\n"
            f"  post-NCCL cfg: {new_result.cfg}\n"
            "Fix: ensure the process group is initialized BEFORE "
            "``post_model_load`` runs so the bootstrap trace captures "
            "real NCCL tables (check cfg.ddp_backend and your launcher "
            "env — torchrun / accelerate launch normally bring the PG "
            "up early). The post-NCCL plan is stashed on "
            "``WrappedModel.post_nccl_search_result`` for inspection."
        )
    else:
        LOG.info(
            "ProTrain: post-NCCL re-run picked the same config; "
            "predicted_iter_s %.4f -> %.4f.",
            wrapped.search_result.predicted_iter_s,
            new_result.predicted_iter_s,
        )
        # Same cfg + block_map: only the cost-model numbers (and the
        # NCCL tables on the trace) refreshed. Safe to publish onto the
        # live fields — the installed runtime still matches.
        wrapped.search_result = new_result
        wrapped._trace = new_trace  # type: ignore[attr-defined]

    return (True, cfg_changed)


def _install_resume_hook(trainer, cfg, wrapped) -> None:
    """Wrap ``trainer._load_from_checkpoint`` so cross-mode resume gathers offloaded chunks before reload."""
    if getattr(trainer, "_protrain_resume_hook_installed", False):
        LOG.debug(
            "ProTrain: resume hook already installed on this trainer; "
            "skipping duplicate patch (idempotent path)."
        )
        return

    original_load = getattr(trainer, "_load_from_checkpoint", None)
    if original_load is None:
        # Test harness without an HF Trainer instance — nothing to patch.
        LOG.debug(
            "ProTrain: trainer has no _load_from_checkpoint attribute; "
            "skipping resume-hook install."
        )
        return

    # Snapshot the optimizer-rebuild hyperparams now so the wrapped
    # closure doesn't have to re-read them off ``trainer.args`` later
    # (Accelerate.prepare may have wrapped the optimizer by then and
    # the hyperparam read becomes ambiguous about which inner optim's
    # values to mirror). Captured as discrete locals (not a kwargs dict)
    # so mypy sees the precise types at the rebuild call site —
    # ``protrain_optimizer_wrapper``'s signature is positional-named
    # with mixed value types (float, tuple[float, float], str | None)
    # and a heterogeneous ``dict[str, object]`` ``**unpack`` flunks
    # type-narrowing.
    args = trainer.args
    rebuild_lr = float(args.learning_rate)
    rebuild_betas = (float(args.adam_beta1), float(args.adam_beta2))
    rebuild_eps = float(args.adam_epsilon)
    rebuild_weight_decay = float(args.weight_decay)
    rebuild_optimizer_name = _resolve_optimizer_name(args, cfg)

    def _patched(resume_from_checkpoint, model=None) -> None:
        # Resolve the chunk manager LAZILY: by the time the patched
        # method fires the wrapper is fully constructed (post_model_load
        # ran), but at install time (post_trainer_create) the
        # chunk_manager attribute IS already present — read it through
        # ``wrapped`` so a future reorder can't strand the closure.
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        if chunk_manager is None:
            LOG.debug(
                "ProTrain resume hook: wrapped.chunk_manager is None; "
                "delegating to the original _load_from_checkpoint."
            )
            return original_load(resume_from_checkpoint, model)

        # Detection: does the chunk manager actually have offloaded
        # chunks live right now? Both ``_cpu_slots`` and
        # ``_chunk_shards`` are populated by ``materialize_offload``;
        # neither is populated under Mode A / all-persistent. Check
        # both so the gate covers replicated AND sharded offload.
        has_offload = bool(
            getattr(chunk_manager, "_cpu_slots", None)
            or getattr(chunk_manager, "_chunk_shards", None)
        )
        if not has_offload:
            LOG.debug(
                "ProTrain resume hook: chunk manager has no offloaded "
                "state (Mode A / all-persistent); delegating to the "
                "original _load_from_checkpoint."
            )
            return original_load(resume_from_checkpoint, model)

        LOG.info(
            "ProTrain resume hook: gathering %d non-persistent chunk(s) "
            "to GPU for cross-mode load_adapter (PEFT load_state_dict "
            "needs full-shape destination tensors).",
            len(getattr(chunk_manager, "_cpu_slots", {}) or {})
            + len(getattr(chunk_manager, "_chunk_shards", {}) or {}),
        )

        # Tear down the CPU adapter before restore_to_gpu invalidates the shard views it holds.
        cpu_optim = getattr(chunk_manager, "cpu_optim", None)
        if cpu_optim is not None:
            try:
                cpu_optim.shutdown()
            except Exception:  # noqa: BLE001 — fail closed
                LOG.exception(
                    "ProTrain resume hook: cpu_optim.shutdown failed; "
                    "aborting before restore_to_gpu invalidates shard views."
                )
                raise
            chunk_manager.cpu_optim = None
        # Drop the GPU adapter ref too — we'll rebuild it after the
        # load. Persistent params keep their data across restore_to_gpu
        # (only standalone-GPU rebind happens), but the GPU adapter's
        # ``param_groups`` dict references the same Parameter instances
        # so the rebuild closes the loop cleanly.
        chunk_manager.gpu_optim = None

        # Step 2: restore_to_gpu rebinds every param.data to standalone
        # GPU storage at full shape. After this, model.load_adapter's
        # PEFT load_state_dict sees real shapes and the size-mismatch
        # error class is gone.
        try:
            chunk_manager.restore_to_gpu()
        except Exception:
            LOG.exception(
                "ProTrain resume hook: chunk_manager.restore_to_gpu "
                "failed; the cross-mode resume cannot proceed. Re-"
                "raising — the alternative (running load against the "
                "zeroed param.data slots) would crash inside HF's load "
                "with the same shape-mismatch error this hook exists "
                "to prevent."
            )
            raise

        # Step 3: run the original load. HF's _load_from_checkpoint
        # signature varies across transformers versions; we forward
        # ``model`` only when it was provided (to match the both-sides
        # signature in transformers/trainer.py:3280).
        if model is None:
            original_load(resume_from_checkpoint)
        else:
            original_load(resume_from_checkpoint, model)

        # Step 4: re-build the offload state. ``materialize_offload``
        # reads ``param.data`` (now the freshly-loaded weights from
        # the checkpoint) and copies into newly-allocated pinned
        # pools, then resets ``param.data`` to the empty placeholder
        # — restoring the same offload contract the wrapper installed
        # at post_model_load time. Idempotency: not relevant here
        # because ``restore_to_gpu`` cleared ``_cpu_slots`` /
        # ``_cpu_param_pool``, so the materialize check passes.
        try:
            chunk_manager.materialize_offload()
        except Exception:
            LOG.exception(
                "ProTrain resume hook: chunk_manager.materialize_offload "
                "failed after the resume load; runtime is now in an "
                "inconsistent state (params on standalone GPU storage "
                "but no offload pinned pool). Re-raising."
            )
            raise

        # Step 5: rebuild the optimizer adapters. The cpu_optim refs
        # into the OLD pinned region were dropped in step 1; the GPU
        # adapter held no chunk-manager-internal refs. A fresh wrap
        # via ``protrain_optimizer_wrapper`` constructs adapters
        # against the NEW pinned pool's ``shard_param`` views and
        # against the (unchanged-identity) persistent ``Parameter``
        # objects.
        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        try:
            new_optim = protrain_optimizer_wrapper(
                wrapped,
                lr=rebuild_lr,
                betas=rebuild_betas,
                eps=rebuild_eps,
                weight_decay=rebuild_weight_decay,
                optimizer_name=rebuild_optimizer_name,
            )
        except Exception:
            LOG.exception(
                "ProTrain resume hook: protrain_optimizer_wrapper rebuild "
                "failed after materialize_offload; runtime can't continue "
                "without an optimizer. Re-raising."
            )
            raise

        # ``trainer.optimizer`` was the pre-resume ``_ProTrainOptimizer``
        # facade. Replace it in-place. Accelerate.prepare hasn't run yet
        # (it runs in _inner_training_loop, downstream of train()'s
        # _load_from_checkpoint call site at transformers/trainer.py
        # ~1413), so the swap is safe — there is no upstream wrapper
        # we'd be invalidating.
        trainer.optimizer = new_optim
        LOG.info(
            "ProTrain resume hook: optimizer adapter rebuilt and "
            "installed on trainer.optimizer; cross-mode resume complete."
        )

    trainer._load_from_checkpoint = _patched  # type: ignore[method-assign]
    trainer._protrain_resume_hook_installed = True  # type: ignore[attr-defined]
    LOG.debug(
        "ProTrain: cross-mode resume hook installed on trainer._load_from_checkpoint"
    )


def _resolve_optimizer_name(args, cfg) -> str | None:
    """Return the optimizer name, preferring HF ``args.optim`` over ``cfg.optimizer``."""
    optimizer_name = getattr(args, "optim", None) or getattr(cfg, "optimizer", None)
    if optimizer_name is not None and not isinstance(optimizer_name, str):
        optimizer_name = getattr(optimizer_name, "value", str(optimizer_name))
    return optimizer_name


def _is_plugin_active(cfg) -> bool:
    """Return True iff both the plugin is registered and auto_memory is on.

    Matches the enable-gate documented on ``ProTrainArgs.protrain_auto_memory``
    and mirrors the ``LigerPlugin`` pattern of reading ``cfg.*`` attributes
    without touching Axolotl-internal state.

    Activation is strictly opt-in: the ``plugins:`` config list must contain
    the canonical ProTrain entry point. Membership is delegated to
    :func:`axolotl.integrations.protrain.args._has_protrain_plugin` so the
    runtime gate cannot drift from the Pydantic validators in ``args.py`` —
    both call sites share ``_PROTRAIN_PLUGIN_KEYS`` as the single source of
    truth. Substring matches such as ``"my-protrain-extension"`` or
    ``"protrain_disabled"`` are intentionally rejected to prevent accidental
    activation.
    """
    if not getattr(cfg, "protrain_auto_memory", False):
        return False
    plugins = getattr(cfg, "plugins", None) or []
    return _has_protrain_plugin(plugins)


def _build_hardware_profile(cfg):
    """Construct a ``HardwareProfile`` from the first visible CUDA device.

    Thin cfg-aware wrapper around
    :func:`axolotl.integrations.protrain.api.hardware.build_hardware_profile`
    — the shared helper that the direct API entry point
    (:func:`axolotl.integrations.protrain.api.auto_wrap`) also calls.

    The plugin-specific work this layer adds is the
    ``zero3_shard`` auto-detect: when no explicit
    ``protrain_zero3_shard`` override is set in YAML, enable sharding
    iff ``world_size > 1`` AND ``protrain_force_all_persistent`` is
    False. The wrapper itself re-checks this (honouring a live
    ``torch.distributed`` process group) and will update the field in
    place — this initial population keeps the cost model honest even
    when the wrapper is bypassed.
    """
    # Resolve world_size first so the zero3_shard auto-detect below
    # consults the same value the shared helper will stamp into the
    # returned HardwareProfile.
    from axolotl.integrations.protrain.api.hardware import _resolve_world_size

    world_size = _resolve_world_size()

    # Mirror protrain_model_wrapper's zero3_shard auto-detect so the
    # searcher's CPU-footprint accounting lines up with the runtime's
    # actual per-rank pinned-memory layout.
    force_all_persistent = bool(getattr(cfg, "protrain_force_all_persistent", False))
    explicit = getattr(cfg, "protrain_zero3_shard", None)
    if explicit is None:
        zero3_shard = (world_size > 1) and (not force_all_persistent)
    else:
        zero3_shard = bool(explicit) and (world_size > 1)

    return _shared_build_hardware_profile(
        world_size_override=world_size,
        zero3_shard=zero3_shard,
    )


class ProTrainPlugin(BasePlugin):
    """Plugin for ProTrain integration with Axolotl.

    Paper: MLSys 2026, arXiv 2406.08334. Exposes:

    * ``get_input_args`` — dotted path to ``ProTrainArgs``.
    * ``post_model_load`` — builds ``HardwareProfile``, calls
      ``protrain_model_wrapper``, stashes the returned ``WrappedModel``
      on ``cfg._protrain_wrapped`` for ``post_trainer_create`` to pick up.
    * ``create_optimizer`` — returns the ``_ProTrainOptimizer`` facade
      constructed from the stashed ``WrappedModel``. Per BasePlugin
      contract, but NOT the wiring path — Axolotl's ``OptimizerMixin``
      does not currently dispatch to ``PluginManager.create_optimizer``,
      so actual optimizer install happens in ``post_trainer_create``.
    * ``post_trainer_create`` — installs ``_ProTrainOptimizer`` on
      ``trainer.optimizer`` directly (this is the real wiring). Also
      auto-detects DDP composition and flips
      ``skip_internal_grad_reduce``.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.protrain.args.ProTrainArgs"

    def get_training_args(self, cfg):
        """Gate ``save_only_model`` on whether ProTrain owns the optim shard.

        Default: ``save_only_model=True``, which skips HF's
        ``_save_optimizer_and_scheduler`` AND ``_save_rng_state``. Real
        save/load of the optimizer goes through the ProTrain checkpoint
        callback (CHECKPOINT_DESIGN.md), not HF's optimizer.pt path —
        ``_ProTrainOptimizer.state_dict`` / ``load_state_dict`` are
        patched to no-ops to coexist with Accelerate's ``prepare``
        round-trip.

        When ``protrain_save_optimizer_state=True`` we flip to
        ``save_only_model=False`` so HF writes ``scheduler.pt`` and
        ``rng_state.pth`` (both needed for a full resume — the ProTrain
        shard only covers the optimizer adam state). HF will also write
        a small ``optimizer.pt`` containing the patched-empty state
        shell; that file is unused on load (the patched
        ``load_state_dict`` is also a no-op) but the I/O cost is
        negligible for the resume completeness it buys.
        """
        if not _is_plugin_active(cfg):
            return None
        save_optim_state = bool(getattr(cfg, "protrain_save_optimizer_state", False))
        return {"save_only_model": not save_optim_state}

    def post_model_load(self, cfg, model: "nn.Module") -> None:
        """Wrap the post-adapter model with the ProTrain runtime.

        Silently no-ops when the plugin is inactive (see
        ``_is_plugin_active``). Called after LoRA adapters are attached
        so persistent-chunk sizing reflects the trainable surface.

        Item 6 — Preflight NCCL measurement. Before invoking
        :func:`protrain_model_wrapper` we attempt to bring the
        ``torch.distributed`` process group up via
        :func:`_early_init_dist_for_nccl` so the profiler trace captures
        real NCCL gather/reduce timings on the live PG (paper §3.3).
        Skipped on single-rank, on non-default ``cfg.ddp_backend``, on
        non-CUDA hosts, and when the PG is already initialised.
        """
        if not _is_plugin_active(cfg):
            return

        # Idempotency: ``post_model_load`` may fire more than once in
        # some test harness configurations (re-runnable trainer
        # bootstrap). The wrapper itself is cheap-but-not-free to repeat
        # (re-measurement, allocator churn) and re-running it would
        # invalidate the chunk-manager handles already stashed on cfg.
        # Compare the stashed wrapper's wrapped model identity against
        # the incoming ``model``: if a DIFFERENT model instance is
        # being loaded (e.g., a test rebuilds the trainer from scratch
        # against a fresh model on the same cfg), the previous wrapper
        # state is stale and must NOT be reused — clear it and proceed
        # with a fresh wrap. Same-model re-entry remains a no-op.
        existing = getattr(cfg, "_protrain_wrapped", None)
        if existing is not None:
            existing_model = getattr(existing, "model", None)
            if existing_model is model:
                LOG.debug(
                    "ProTrain: post_model_load called with _protrain_wrapped "
                    "already populated for the same model; skipping re-wrap "
                    "(idempotent path)."
                )
                return
            LOG.warning(
                "ProTrain: post_model_load called with _protrain_wrapped "
                "populated for a DIFFERENT model instance; clearing the "
                "stale wrapper and re-wrapping. (Test harness or "
                "re-trainer-build path.)"
            )
            # Cascade the canonical teardown so the stale wrapper releases
            # pinned-host pools, joins the CPU-Adam worker thread, and
            # removes hooks before we drop the reference. Without this
            # the dropped wrapper relies on Python GC to fire __del__ on
            # ChunkManager / CpuFusedAdamAdapter, which can run too late
            # to keep the next wrap's allocator math honest.
            try:
                existing.close()
            except Exception as exc:  # noqa: BLE001
                # Fail closed; swallowing leaks pinned pools, hooks, and the CPU-Adam worker into the next wrap.
                LOG.exception(
                    "ProTrain: stale-wrapper close() failed during re-wrap; "
                    "aborting to avoid leaking pinned pools / Adam worker / "
                    "tensor hooks across the next wrap."
                )
                raise RuntimeError(
                    "ProTrain failed to close the previous wrapped model "
                    "during re-wrap; aborting to keep teardown deterministic."
                ) from exc
            cfg._protrain_wrapped = None  # type: ignore[attr-defined]

        from axolotl.integrations.protrain.api import protrain_model_wrapper

        # Bring up dist.init *before* building the hardware profile so
        # ``_build_hardware_profile`` can report the true world size and
        # ``protrain_model_wrapper.run_trace`` (which calls
        # ``measure_nccl`` internally) sees the live PG.
        _early_init_dist_for_nccl(cfg)

        # ---- Compute target device for the wrapper (hint, no move) -----
        # Per-rank target device that downstream GPU allocations
        # (BufferPool, ChunkManager, profiler) should use, computed
        # exclusively from ``LOCAL_RANK`` + visible CUDA device count.
        # We do NOT call ``model.to(target)`` here — eagerly
        # materializing the full model on a single device defeats the
        # ProTrain chunk-offload promise (paper §3.1) and the failed-
        # ``to()`` rescue path that previously swallowed ``RuntimeError``
        # was a footgun: it left the wrapper running on a still-CPU
        # model whose downstream ``next(model.parameters()).device``
        # read would seed every GPU-side allocation against the wrong
        # device. Instead, the target is threaded through to
        # ``protrain_model_wrapper`` as an explicit kwarg; the wrapper
        # takes responsibility for placement under its own OOM-aware
        # path. Model-mapped (``hf_device_map``) loads are handled by
        # passing ``target_device=None`` so the wrapper falls back to
        # the model's existing device-map placement.
        import os as _os

        target_device = None
        try:
            import torch as _torch
        except ImportError:
            _torch = None  # type: ignore[assignment]
        if _torch is not None and _torch.cuda.is_available():
            # Skip on device-mapped (``accelerate``-dispatched) loads.
            # The device map already pins each shard to a CUDA ordinal,
            # so the wrapper inherits the right placement from the
            # model's parameters; computing a single target_device
            # would either be wrong (forces collapse) or redundant.
            hf_device_map = getattr(model, "hf_device_map", None)
            if hf_device_map:
                LOG.info(
                    "ProTrain: model has hf_device_map=%s; deferring "
                    "device selection to the wrapper (target_device=None).",
                    hf_device_map,
                )
            else:
                # Defensive parse: a non-numeric LOCAL_RANK would raise
                # here and abort plugin init before the safer fallback
                # in _build_hardware_profile() runs; a negative would
                # slip through as cuda:-1. Mirror the same try/except +
                # range guard used at _build_hardware_profile().
                raw_local_rank = _os.environ.get("LOCAL_RANK", "0")
                try:
                    local_rank = int(raw_local_rank)
                except ValueError:
                    LOG.warning(
                        "ProTrain: invalid LOCAL_RANK=%r; falling back to current CUDA device.",
                        raw_local_rank,
                    )
                    local_rank = _torch.cuda.current_device()
                visible = _torch.cuda.device_count()
                if 0 <= local_rank < visible:
                    target_device = _torch.device("cuda", local_rank)
                    # Stash on the model as a lightweight metadata hint
                    # so downstream callers that bypass the kwarg path
                    # (e.g. plugin-less direct callers reaching the
                    # wrapper through a third-party harness) can still
                    # read the intended target.
                    try:
                        model._protrain_target_device = target_device  # type: ignore[attr-defined]
                    except (AttributeError, TypeError):
                        # Frozen / __slots__ models — the kwarg path
                        # below is the canonical handoff anyway.
                        pass
                    LOG.info(
                        "ProTrain: target_device=%s (LOCAL_RANK=%d, visible=%d) — "
                        "actual placement deferred to the wrapper / Accelerate.prepare.",
                        target_device,
                        local_rank,
                        visible,
                    )
                else:
                    LOG.warning(
                        "ProTrain: CUDA available but LOCAL_RANK=%d is out of "
                        "range for visible device count %d (CUDA_VISIBLE_DEVICES "
                        "masking?); leaving target_device unset, the wrapper will "
                        "infer from the model's current placement.",
                        local_rank,
                        visible,
                    )

        hw = _build_hardware_profile(cfg)

        # Pull knobs / overrides off the merged cfg. Pydantic already
        # validated the mutex with deepspeed/fsdp; here we just read.
        micro_batch_size = int(getattr(cfg, "micro_batch_size", 1) or 1)
        seq_len = int(getattr(cfg, "sequence_len", 1024) or 1024)
        capacity_bytes = getattr(cfg, "protrain_capacity_bytes", None)
        cpu_capacity_bytes = getattr(cfg, "protrain_cpu_capacity_bytes", None)
        cache_dir = getattr(cfg, "protrain_cache_dir", None)
        force_all_persistent = bool(
            getattr(cfg, "protrain_force_all_persistent", False)
        )

        n_persist_override = getattr(cfg, "protrain_n_persist_override", None)
        n_buffer_override = getattr(cfg, "protrain_n_buffer_override", None)
        n_swap_override = getattr(cfg, "protrain_n_swap_override", None)
        n_checkpoint_override = getattr(cfg, "protrain_n_checkpoint_override", None)
        n_offload_override = getattr(cfg, "protrain_n_offload_override", None)
        zero3_shard = getattr(cfg, "protrain_zero3_shard", None)

        # auto_mode defaults to True (see ProTrainArgs). On the auto
        # path, the wrapper runs the searcher first and then calls
        # :func:`axolotl.integrations.protrain.api.model_wrapper._select_mode`
        # to resolve ``(force_all_persistent, zero3_shard)`` from
        # workload fit + CPU-RAM-per-rank. When explicitly disabled,
        # the wrapper honours the user's flags verbatim — see the
        # ProTrainArgs docstrings for the override semantics.
        auto_mode = getattr(cfg, "protrain_auto_mode", True)
        if auto_mode is None:
            auto_mode = True

        wrapped = protrain_model_wrapper(
            model,
            model_config=getattr(model, "config", None),
            hardware_profile=hw,
            batch_size=micro_batch_size,
            seq_len=seq_len,
            capacity_bytes=capacity_bytes,
            cpu_capacity_bytes=cpu_capacity_bytes,
            cache_dir=cache_dir,
            force_all_persistent=force_all_persistent,
            n_persist_override=n_persist_override,
            n_buffer_override=n_buffer_override,
            n_swap_override=n_swap_override,
            n_checkpoint_override=n_checkpoint_override,
            n_offload_override=n_offload_override,
            zero3_shard=zero3_shard,
            auto_mode=bool(auto_mode),
            target_device=target_device,
        )

        # Stash on cfg so post_trainer_create (which only receives cfg +
        # trainer) can recover the WrappedModel. Using a leading
        # underscore to signal "runtime state, not YAML-serialisable".
        cfg._protrain_wrapped = wrapped  # type: ignore[attr-defined]

        picked = wrapped.search_result.cfg
        # Derive the effective-mode string from the chunk manager's
        # post-wrapper state rather than the raw user flag: with
        # ``auto_mode=True`` the selector may have overridden the
        # user's force_all_persistent / zero3_shard intent, and the
        # log should reflect what's actually installed.
        chunk_manager = cast("ChunkManager", wrapped.chunk_manager)
        n_chunk_total = getattr(chunk_manager.layout, "N_chunk", -1)
        effective_force_persistent = int(picked.n_persist) >= int(n_chunk_total)
        effective_zero3 = bool(getattr(chunk_manager, "zero3_shard", False))
        LOG.info(
            "ProTrain: %s config picked (n_persist=%d, n_buffer=%d, "
            "n_checkpoint=%d, force_all_persistent=%s, zero3_shard=%s, "
            "auto_mode=%s)",
            type(getattr(model, "base_model", model)).__name__,
            getattr(picked, "n_persist", -1),
            getattr(picked, "n_buffer", -1),
            getattr(picked, "n_checkpoint", -1),
            effective_force_persistent,
            effective_zero3,
            bool(auto_mode),
        )

    def create_optimizer(self, cfg, trainer: "Trainer") -> "Optimizer | None":
        """Return the ProTrain optimizer facade, or ``None`` when inactive."""
        if not _is_plugin_active(cfg):
            return None

        wrapped = getattr(cfg, "_protrain_wrapped", None)
        if wrapped is None:
            # post_model_load wasn't called (or the model was None) —
            # fall through to Axolotl's default optimizer path rather
            # than raise, since that matches every other plugin's
            # "inactive -> return None" contract.
            LOG.warning(
                "ProTrain.create_optimizer: no _protrain_wrapped on cfg; "
                "post_model_load must have been skipped. Falling through to "
                "the default optimizer."
            )
            return None

        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        args = trainer.args
        lr = float(args.learning_rate)
        betas = (float(args.adam_beta1), float(args.adam_beta2))
        eps = float(args.adam_epsilon)
        weight_decay = float(args.weight_decay)
        # Forward the optimizer name so the wrapper can route 8-bit-bnb to GpuAdamW8bitAdapter.
        optimizer_name = getattr(args, "optim", None) or getattr(cfg, "optimizer", None)
        if optimizer_name is not None and not isinstance(optimizer_name, str):
            optimizer_name = getattr(optimizer_name, "value", str(optimizer_name))

        LOG.info(
            "ProTrain.create_optimizer: lr=%.3e betas=%s eps=%.1e wd=%.3e optimizer=%s",
            lr,
            betas,
            eps,
            weight_decay,
            optimizer_name,
        )

        return protrain_optimizer_wrapper(
            wrapped,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
        )

    def post_trainer_create(self, cfg, trainer: "Trainer") -> None:
        """Install the ProTrain optimizer on the trainer.

        Axolotl's ``OptimizerMixin.create_optimizer`` does not dispatch
        to ``PluginManager.create_optimizer`` (unlike
        ``SchedulerMixin.create_scheduler``), so relying on
        :meth:`create_optimizer` alone leaves the plugin inert and the
        trainer falls back to vanilla AdamW. HuggingFace ``Trainer``
        checks ``self.optimizer`` before rebuilding one — setting
        ``trainer.optimizer`` here intercepts that path.

        Also auto-detects DDP composition and flips
        ``chunk_manager.skip_internal_grad_reduce`` so the outer DDP
        wrapper owns the cross-rank grad all-reduce rather than fighting
        with ProTrain's per-chunk reduce.
        """
        if not _is_plugin_active(cfg):
            return

        # Idempotency: ``post_trainer_create`` may fire more than once on
        # re-entrant trainer bootstraps (test harness re-creates, or a
        # caller manually re-running the hook). Reinstalling stacks
        # duplicate save/load hooks and double-registers the checkpoint
        # callback — guard so a second invocation is a debug-logged
        # no-op.
        if getattr(trainer, "_protrain_post_trainer_create_done", False):
            LOG.debug(
                "ProTrain: post_trainer_create already ran on this trainer; "
                "skipping duplicate install (idempotent path)."
            )
            return

        wrapped = getattr(cfg, "_protrain_wrapped", None)
        if wrapped is None:
            LOG.warning(
                "ProTrain: post_trainer_create fired without wrapped model; "
                "skipping optimizer install. post_model_load must have been "
                "skipped (non-CUDA run?) — falling back to the default "
                "optimizer."
            )
            return

        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        args = trainer.args
        optimizer_name = getattr(args, "optim", None) or getattr(cfg, "optimizer", None)
        if optimizer_name is not None and not isinstance(optimizer_name, str):
            optimizer_name = getattr(optimizer_name, "value", str(optimizer_name))
        optim = protrain_optimizer_wrapper(
            wrapped,
            lr=float(args.learning_rate),
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_epsilon),
            weight_decay=float(args.weight_decay),
            optimizer_name=optimizer_name,
        )

        # ``_ProTrainOptimizer.state_dict`` / ``load_state_dict`` already
        # implement the empty-shell + discard-payload behavior that HF
        # Trainer and Accelerate need at ``prepare`` time (see
        # ``api/optim_wrapper.py``). The bring-up path that previously
        # monkey-patched these methods on the instance was redundant once
        # the class implementations landed.
        trainer.optimizer = optim
        LOG.info(
            "ProTrain: installed protrain_optimizer_wrapper on trainer.optimizer "
            "(lr=%.3e betas=%s eps=%.1e wd=%.3e)",
            float(args.learning_rate),
            (float(args.adam_beta1), float(args.adam_beta2)),
            float(args.adam_epsilon),
            float(args.weight_decay),
        )

        # Patch _load_from_checkpoint: restore_to_gpu before load (offloaded LoRA factors are size (0,)) then re-offload + rebuild optim.
        _install_resume_hook(trainer, cfg, wrapped)

        # ---- Optimizer-state checkpoint/resume (CHECKPOINT_DESIGN.md) ----
        # Opt-in via protrain_save_optimizer_state. The save side is a
        # TrainerCallback (on_save fires after HF writes its standard
        # checkpoint dir); the load side is a monkey-patch on
        # _load_optimizer_and_scheduler (HF has no on_load_checkpoint
        # callback, and on_train_begin fires after the load slot).
        if bool(getattr(cfg, "protrain_save_optimizer_state", False)):
            from axolotl.integrations.protrain.api.checkpoint import (
                DEFAULT_SAVE_MAX_BYTES,
                install_load_hook,
                make_checkpoint_callback,
            )

            cfg_max = getattr(cfg, "protrain_optim_save_max_bytes", None)
            save_max = int(cfg_max) if cfg_max is not None else DEFAULT_SAVE_MAX_BYTES
            verify_replicated = bool(
                getattr(cfg, "protrain_save_optim_verify_replicated", False)
            )
            allow_online_reshard = bool(
                getattr(cfg, "protrain_allow_online_reshard", False)
            )
            trainer.add_callback(
                make_checkpoint_callback(
                    save_max_bytes=save_max,
                    verify_replicated=verify_replicated,
                )
            )
            install_load_hook(trainer, optim, allow_online_reshard=allow_online_reshard)
            LOG.info(
                "ProTrain: optimizer-state checkpointing enabled "
                "(save_max_bytes=%d ~= %.2f GiB, verify_replicated=%s, "
                "allow_online_reshard=%s). "
                "Save side: ProTrainOptimizerCheckpointCallback. "
                "Load side: trainer._load_optimizer_and_scheduler patched.",
                save_max,
                save_max / 1024**3,
                verify_replicated,
                allow_online_reshard,
            )

        # ---- DDP composition detection ----------------------------------
        # If the trainer's model is wrapped in DistributedDataParallel,
        # defer cross-rank grad all-reduce to DDP and silence ProTrain's
        # internal reduce. Conversely, surface the case of multi-rank
        # init without DDP so the operator knows ProTrain's own reduce
        # path is still active (which is correct — just unusual).
        try:
            import torch
            from torch.nn.parallel import DistributedDataParallel
        except ImportError:
            return

        is_ddp = isinstance(trainer.model, DistributedDataParallel) or (
            hasattr(trainer, "model_wrapped")
            and isinstance(
                getattr(trainer, "model_wrapped", None), DistributedDataParallel
            )
        )
        if is_ddp:
            # DDP composition is incompatible with ZeRO-3 sharding —
            # ``skip_internal_grad_reduce=True`` only suppresses the
            # PERSISTENT-chunk all-reduce path; non-persistent sharded
            # chunks still route through
            # ``ChunkManager._reduce_scatter_and_offload_shard``
            # unconditionally whenever ``_chunk_shards`` has entries.
            # With DDP's bucketed all-reduce ALSO firing on every
            # parameter, gradients double-synchronize and the effective
            # update is corrupted. At this point materialize_offload
            # has already created per-rank shards, so we cannot cleanly
            # revert here — hard-raise so the operator fixes the
            # configuration before training starts.
            chunk_manager = cast("ChunkManager", wrapped.chunk_manager)
            if getattr(chunk_manager, "zero3_shard", False):
                raise RuntimeError(
                    "ProTrain: DDP wrapping detected with active "
                    "zero3_shard=True. Non-persistent sharded chunks call "
                    "reduce_scatter via "
                    "ChunkManager._reduce_scatter_and_offload_shard while "
                    "DDP also issues bucketed all-reduce on every parameter "
                    "— gradients double-synchronize and the effective "
                    "update is corrupted (skip_internal_grad_reduce only "
                    "silences the persistent-chunk path, not the sharded "
                    "reduce_scatter). Either (a) rebuild the runtime in "
                    "replicated mode by setting "
                    "``protrain_zero3_shard: false`` in YAML before "
                    "training, or (b) disable DDP wrapping (e.g. by "
                    "removing DDP from the trainer config) and let "
                    "ProTrain own grad reduction."
                )
            chunk_manager.skip_internal_grad_reduce = True
            LOG.info(
                "ProTrain: detected DDP composition; set "
                "skip_internal_grad_reduce=True (DDP owns the cross-rank grad "
                "all-reduce)"
            )
        elif (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            LOG.warning(
                "ProTrain: multi-rank init (world_size=%d) detected but "
                "trainer.model is not wrapped in DistributedDataParallel; "
                "ProTrain's internal per-chunk grad all-reduce path remains "
                "active. This is the correct path for non-DDP multi-rank "
                "runs, but surface it here because it is unusual.",
                torch.distributed.get_world_size(),
            )

        # Re-measure NCCL now that dist is up. No-op on single rank or
        # when the trace already has populated tables.
        _remeasure_nccl_and_research(wrapped)

        # Mark this trainer as fully bootstrapped so a re-entrant call
        # to ``post_trainer_create`` short-circuits at the guard above
        # rather than stacking duplicate optimizer / load-hook /
        # checkpoint-callback registrations.
        trainer._protrain_post_trainer_create_done = True  # type: ignore[attr-defined]


__all__ = ["ProTrainPlugin"]
