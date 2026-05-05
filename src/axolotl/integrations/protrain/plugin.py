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
from axolotl.integrations.protrain.args import _has_protrain_plugin
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from transformers import Trainer

    from axolotl.integrations.protrain.chunk import ChunkManager

LOG = get_logger(__name__)


# Default PCIe H2D bandwidth assumed for HardwareProfile construction when
# no measured value is available. 13 GB/s matches a typical PCIe Gen4 x16
# 3090 rig; the profiler's microbench will overwrite this once the cache
# key misses and a full profile runs — this constant only seeds the
# constructor for the cost model's effective-bandwidth prior.
_DEFAULT_PCIE_BPS = 13e9


def _resolve_world_size_from_env() -> int:
    """Return ``WORLD_SIZE`` from the env, defaulting to 1.

    Both torchrun and Accelerate's launchers populate ``WORLD_SIZE`` /
    ``RANK`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT`` before
    the user script starts. We treat the env as the source of truth here
    because the plugin's ``post_model_load`` runs before the trainer (and
    thus before Accelerate) has had a chance to call
    :func:`torch.distributed.init_process_group`.
    """
    import os

    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


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
    ``accelerate/state.py`` ``PartialState.__init__`` lines 219–244.

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
        save_cached_trace(new_key, new_trace)
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
        # inputs. Hitting this branch implies either the early init was
        # skipped (custom backend, single-rank → multi-rank weirdness)
        # or the late path is plumbed against a different PG. Logged at
        # DEBUG since it's expected-rare under the new flow; bump to
        # INFO/WARN locally if you're debugging the late-bind path.
        LOG.debug(
            "ProTrain: post-NCCL search picked a different config than "
            "the bootstrap prediction. cfg %s -> %s; stashing the "
            "post-NCCL plan on WrappedModel.post_nccl_search_result for "
            "telemetry and LEAVING search_result/_trace untouched so "
            "they continue to reflect the installed runtime "
            "(chunk_manager / scheduler / hooks are already wired for "
            "the bootstrap config; the optimizer state slots ride on "
            "those, so we cannot rebuild mid-flight). The running step "
            "uses the bootstrap config; future runs will hit the "
            "multi-rank cache and pick the new config from the start. "
            "Reaching this branch suggests early dist init was skipped "
            "— check cfg.ddp_backend / launcher env.",
            wrapped.search_result.cfg,
            new_result.cfg,
        )
        # Telemetry-only: keep the late-search outputs visible to
        # callers (tests, dynamic re-tuning) without overwriting the
        # live runtime contract reported via ``search_result``/``_trace``.
        wrapped.post_nccl_search_result = new_result  # type: ignore[attr-defined]
        wrapped.post_nccl_trace = new_trace  # type: ignore[attr-defined]
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

    Populates ``zero3_shard`` from the same auto-detect logic used by
    :func:`protrain_model_wrapper`: when no explicit
    ``protrain_zero3_shard`` override is set in YAML, enable sharding
    iff ``world_size > 1`` AND ``protrain_force_all_persistent`` is
    False. The wrapper itself re-checks this (honouring a live
    ``torch.distributed`` process group) and will update the field in
    place — this initial population keeps the cost model honest even
    when the wrapper is bypassed.
    """
    import torch

    from axolotl.integrations.protrain.types import HardwareProfile

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ProTrain plugin requires a CUDA device; torch.cuda.is_available() is False."
        )

    # Honour CUDA_VISIBLE_DEVICES — the ordinal here is logical, which
    # resolves to whatever the user masked in via the env var. Read this
    # rank's device (set by ``torch.cuda.set_device(LOCAL_RANK)`` in
    # ``post_model_load``) so heterogeneous-memory multi-GPU rigs report
    # the correct ``capacity_bytes`` / SKU per rank instead of always
    # reading device 0.
    import os

    raw_local_rank = os.environ.get("LOCAL_RANK", "0")
    try:
        local_rank = int(raw_local_rank)
    except ValueError:
        LOG.warning(
            "ProTrain: invalid LOCAL_RANK=%r; falling back to current CUDA device.",
            raw_local_rank,
        )
        local_rank = torch.cuda.current_device()

    visible = int(torch.cuda.device_count())
    if visible <= 0:
        raise RuntimeError("ProTrain plugin requires at least one visible CUDA device.")
    if not (0 <= local_rank < visible):
        LOG.warning(
            "ProTrain: LOCAL_RANK=%d out of visible CUDA range [0, %d); "
            "falling back to current CUDA device.",
            local_rank,
            visible,
        )
        device = torch.cuda.current_device()
    else:
        device = local_rank
    props = torch.cuda.get_device_properties(device)
    gpu_memory_bytes = int(props.total_memory)
    gpu_sku = torch.cuda.get_device_name(device)

    # Measured PCIe bandwidth lives in the profiler trace; at plugin load
    # time we seed a reasonable prior. The cost model uses hardware_profile
    # for effective-bandwidth derating (cost/bandwidth.py) where the
    # absolute value matters less than the ratio against n_swap traffic.
    pcie_h2d_bps = _DEFAULT_PCIE_BPS
    pcie_d2h_bps = _DEFAULT_PCIE_BPS

    # Prefer the live process group when one is up (set by our early
    # init in ``post_model_load`` for multi-rank torchrun runs). Fall
    # back to ``WORLD_SIZE`` env (also accurate under torchrun, defaults
    # to 1 for single-process runs). Do NOT use ``torch.cuda.device_count()``
    # as a fallback: visible GPU count is not the distributed rank count,
    # so on a single-process run on a multi-GPU host this would inflate
    # ``world_size`` from 1 to N and skew the profiler cache key, the
    # per-rank CPU-capacity budget, and the cost-model sharding divisor
    # before the wrapper has a chance to correct it.
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            world_size = max(1, int(_dist.get_world_size()))
        else:
            world_size = _resolve_world_size_from_env()
    except ImportError:
        world_size = 1

    # Mirror protrain_model_wrapper's zero3_shard auto-detect so the
    # searcher's CPU-footprint accounting lines up with the runtime's
    # actual per-rank pinned-memory layout.
    force_all_persistent = bool(getattr(cfg, "protrain_force_all_persistent", False))
    explicit = getattr(cfg, "protrain_zero3_shard", None)
    if explicit is None:
        zero3_shard = (world_size > 1) and (not force_all_persistent)
    else:
        zero3_shard = bool(explicit) and (world_size > 1)

    return HardwareProfile(
        gpu_sku=gpu_sku,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=world_size,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        has_nvlink=False,
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
        if getattr(cfg, "_protrain_wrapped", None) is not None:
            LOG.debug(
                "ProTrain: post_model_load called with _protrain_wrapped "
                "already populated; skipping re-wrap (idempotent path)."
            )
            return

        from axolotl.integrations.protrain.api import protrain_model_wrapper

        # Bring up dist.init *before* building the hardware profile so
        # ``_build_hardware_profile`` can report the true world size and
        # ``protrain_model_wrapper.run_trace`` (which calls
        # ``measure_nccl`` internally) sees the live PG.
        _early_init_dist_for_nccl(cfg)

        # ---- Move model to cuda:LOCAL_RANK if needed --------------------
        # ``protrain_model_wrapper`` reads
        # ``next(model.parameters()).device`` to seed the profiler
        # tracker, which calls ``torch.cuda.memory_stats(device)`` —
        # that raises ``ValueError: Expected a cuda device`` when the
        # device is CPU. Under ``accelerate launch`` (the path
        # ``axolotl train`` takes for single-GPU runs), Axolotl's
        # ``choose_device`` deliberately sets ``cfg.device_map = None``
        # when ``ACCELERATE_USE_*`` env vars are present (see
        # ``utils/config/__init__.py``); HF Trainer relies on
        # ``Accelerator.prepare`` later in the bootstrap to move the
        # model. By that point our ``post_model_load`` has already
        # fired with the model still on CPU. The in-process
        # ``axolotl.train.train`` path doesn't hit this because no
        # ``ACCELERATE_USE_*`` env vars are set, so ``device_map`` falls
        # to ``"auto"`` and the model is GPU-resident at load time.
        # We close the gap by moving the model ourselves; idempotent
        # when already on the target device. The gate also catches the
        # case where the model is already on CUDA but on the *wrong*
        # ordinal (e.g. left on ``cuda:0`` while ``LOCAL_RANK=2``) — we
        # pin it to ``cuda:LOCAL_RANK`` so the profiler reads memory
        # stats from the device this rank will actually train on.
        import os as _os

        try:
            import torch as _torch

            current_device = next(model.parameters()).device
        except (StopIteration, ImportError):
            current_device = None
            _torch = None  # type: ignore[assignment]
        if (
            current_device is not None
            and _torch is not None
            and _torch.cuda.is_available()
        ):
            # Defensive parse: a non-numeric LOCAL_RANK would raise here
            # and abort plugin init before the safer fallback in
            # _build_hardware_profile() runs; a negative would slip
            # through as cuda:-1. Mirror the same try/except + range
            # guard used at _build_hardware_profile().
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
            # ``current_device.index`` is ``None`` for a bare
            # ``torch.device("cuda")`` without an explicit ordinal
            # (resolves to the current device at runtime); treat that as
            # "wrong ordinal" so we pin it to ``cuda:LOCAL_RANK``.
            on_wrong_cuda = current_device.type == "cuda" and (
                current_device.index is None or current_device.index != local_rank
            )
            needs_move = current_device.type != "cuda" or on_wrong_cuda
            if not needs_move:
                pass  # already on cuda:local_rank, no-op
            elif 0 <= local_rank < visible:
                target = f"cuda:{local_rank}"
                LOG.info(
                    "ProTrain: model is on %s; moving to %s before wrap "
                    "(post_model_load fired pre-Accelerate.prepare).",
                    current_device,
                    target,
                )
                model.to(target)
            else:
                LOG.warning(
                    "ProTrain: model is on %s and CUDA is available, but "
                    "LOCAL_RANK=%d is out of range for visible device count "
                    "%d (CUDA_VISIBLE_DEVICES masking?); skipping pre-wrap "
                    "model.to() and deferring placement to Accelerate.prepare.",
                    current_device,
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

        LOG.info(
            "ProTrain.create_optimizer: lr=%.3e betas=%s eps=%.1e wd=%.3e",
            lr,
            betas,
            eps,
            weight_decay,
        )

        return protrain_optimizer_wrapper(
            wrapped,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
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
        optim = protrain_optimizer_wrapper(
            wrapped,
            lr=float(args.learning_rate),
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_epsilon),
            weight_decay=float(args.weight_decay),
        )

        # ``_ProTrainOptimizer.state_dict`` raises NotImplementedError
        # (optim-state checkpointing is M6 scope). HF Trainer and
        # Accelerate both call ``state_dict`` unconditionally — HF at
        # checkpoint save (silenced via ``save_only_model=True`` in
        # ``get_training_args``) and Accelerate at ``prepare`` time for
        # device-placement (NOT silenced). Override the two methods on
        # this instance with safe no-ops so the bring-up path survives
        # without having to edit the api/ module (out-of-scope per the
        # fix plan). The safe no-op returns an empty param-state dict
        # preserving HF's ``{"param_groups": ...}`` shape so
        # Accelerate's ``move_to_device(state_dict, ...)`` +
        # ``load_state_dict(state_dict)`` round-trip does not crash.
        def _empty_state_dict(_self=optim):  # type: ignore[misc]
            return {
                "state": {},
                "param_groups": [
                    {k: v for k, v in g.items() if k != "params"}
                    | {"params": [i for i, _ in enumerate(g["params"])]}
                    for g in _self.param_groups
                ],
            }

        def _noop_load_state_dict(_state_dict, _self=optim):  # type: ignore[misc]
            # Accelerate re-loads the same (device-moved) state we just
            # returned — since neither adapter owns persistent state on
            # the torch side, discarding it is safe for the M5 scope.
            return None

        optim.state_dict = _empty_state_dict  # type: ignore[method-assign]
        optim.load_state_dict = _noop_load_state_dict  # type: ignore[method-assign]

        trainer.optimizer = optim
        LOG.info(
            "ProTrain: installed protrain_optimizer_wrapper on trainer.optimizer "
            "(lr=%.3e betas=%s eps=%.1e wd=%.3e)",
            float(args.learning_rate),
            (float(args.adam_beta1), float(args.adam_beta2)),
            float(args.adam_epsilon),
            float(args.weight_decay),
        )

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
