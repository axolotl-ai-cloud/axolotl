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

"""BasePlugin subclass for ProTrain."""

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


# One-shot session flag so the inert-plugin WARN fires at most once per process.
_INERT_AUTO_MEMORY_WARN_FIRED: bool = False


def _maybe_warn_inert_plugin(cfg) -> None:
    """One-time WARN when ProTrain plugin is listed but ``protrain_auto_memory`` is off.

    Plugin runtime hooks (materialize_offload, optimizer wrap, chunk scheduler)
    only fire when both the plugin is registered AND ``protrain_auto_memory: true``;
    listing the plugin alone is a silent no-op and was the root cause of v15-v52's
    inert "measurements" (proposal §6.pp). Fire once per session via a module flag.
    """
    global _INERT_AUTO_MEMORY_WARN_FIRED
    if _INERT_AUTO_MEMORY_WARN_FIRED:
        return
    plugins = getattr(cfg, "plugins", None) or []
    if not _has_protrain_plugin(plugins):
        return
    if getattr(cfg, "protrain_auto_memory", False):
        return
    LOG.warning(
        "ProTrainPlugin is registered in `plugins:` but `protrain_auto_memory: "
        "true` is NOT set in the YAML. The plugin's runtime hooks "
        "(materialize_offload, optimizer wrap, chunk scheduler) will NOT run. "
        "Training will proceed as vanilla axolotl + accelerate — the args "
        "schema is honored but no memory management happens.\n"
        "\n"
        "To activate ProTrain, add `protrain_auto_memory: true` to your YAML. "
        "See proposal §3.4 and §16 PR #9."
    )
    _INERT_AUTO_MEMORY_WARN_FIRED = True


def _early_init_dist_for_nccl(cfg) -> int:
    """Init torch.distributed early so the profiler captures real NCCL times."""
    import os

    world_size = _resolve_world_size_from_env()
    if world_size <= 1:
        return 1

    # Bail without LOCAL_RANK/RANK; manual WORLD_SIZE export without torchrun isn't enough.
    if os.environ.get("LOCAL_RANK") is None or os.environ.get("RANK") is None:
        LOG.warning(
            "ProTrain: WORLD_SIZE=%d but LOCAL_RANK/RANK not set — assuming "
            "non-launcher environment, skipping early dist init. NCCL "
            "tables will be empty and Mode-C selection may be suboptimal.",
            world_size,
        )
        return 1

    # Skip custom backends so Accelerate/HF can own the init.
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
        # Already up; just surface the live world.
        try:
            return int(dist.get_world_size())
        except (RuntimeError, ValueError):
            return world_size

    if not torch.cuda.is_available():
        # NCCL needs CUDA; defer to late-bind path.
        LOG.info(
            "ProTrain: CUDA unavailable; skipping early NCCL dist init "
            "(WORLD_SIZE=%d).",
            world_size,
        )
        return 1

    # Bind LOCAL_RANK GPU before NCCL init so collectives target the per-rank shard.
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
        "world_size=%d, rank=%s, local_rank=%s) so the profiler captures "
        "real NCCL gather/reduce times. Accelerate detects is_initialized "
        "and skips re-init.",
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


def _eager_nccl_warmup(chunk_manager, device) -> None:
    """Fire one no-op of each NCCL collective the per-chunk path uses.

    Why: on consumer non-NVLink topology, lazy ``ncclCommInitRank`` for the
    per-chunk reduce_scatter / all_reduce / OFFLOAD re-gather ring serializes
    on the per-rank CUDA stream during the FIRST training iteration —
    measured at 254-345 s under v71-redux watchdog. Issuing the same
    collective shapes here amortizes the init cost into a deterministic,
    watchdog-measurable init phase BEFORE the first iter's autograd-internal
    dispatch.

    Catches and logs all exceptions: a broken warmup must not block training.
    """
    import time

    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return

    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        world_size = int(dist.get_world_size())
    except (RuntimeError, ValueError):
        return
    if world_size <= 1:
        return

    zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))
    n_chunk = int(getattr(getattr(chunk_manager, "layout", None), "N_chunk", 0) or 0)

    LOG.info(
        "ProTrain: eager NCCL warm-up starting (world_size=%d, "
        "zero3_shard=%s, n_chunk=%d, device=%s)",
        world_size,
        zero3_shard,
        n_chunk,
        device,
    )
    t0 = time.perf_counter()

    try:
        # bf16 matches the per-chunk all_reduce / reduce_scatter dtype used by
        # _coalesced_all_reduce_persistent_grads + _reduce_scatter_and_offload_shard.
        # NCCL's communicator init is one-shot per ProcessGroup; the dtype/shape
        # used to provoke it does not need to match every later op.
        warm = torch.zeros(1, device=device, dtype=torch.bfloat16)
        dist.all_reduce(warm, op=dist.ReduceOp.SUM)

        rs_in = torch.zeros(world_size, device=device, dtype=torch.bfloat16)
        rs_out = torch.zeros(1, device=device, dtype=torch.bfloat16)
        dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM)

        if zero3_shard:
            # _gather_sharded uses uint8 buffers; warm the all_gather entry.
            ag_in = torch.zeros(1, device=device, dtype=torch.uint8)
            ag_out = torch.zeros(world_size, device=device, dtype=torch.uint8)
            dist.all_gather_into_tensor(ag_out, ag_in)

        dist.barrier()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

        elapsed = time.perf_counter() - t0
        LOG.info(
            "ProTrain: eager NCCL warm-up complete in %.2fs "
            "(world_size=%d, zero3_shard=%s)",
            elapsed,
            world_size,
            zero3_shard,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        LOG.warning(
            "ProTrain: eager NCCL warm-up failed after %.2fs (%s: %s); "
            "first training iter will pay the lazy ncclCommInitRank cost "
            "as before. Set protrain_eager_nccl_warmup: false to silence.",
            elapsed,
            type(exc).__name__,
            exc,
        )


def _remeasure_nccl_and_research(wrapped) -> tuple[bool, bool]:
    """Late-bind real NCCL timings into the cached trace, then re-run search()."""
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

    # Idempotency: tables already populated → skip re-measurement. We still defensively
    # verify that the bootstrap search_result.cfg/block_map are identical across ranks,
    # so PR #20-class mixed-SKU TFLOPS divergence (per-rank measure_compute_rate
    # outliers flipping the 1%-noise-band tie-breaker in search/exhaustive.py) surfaces
    # here instead of deadlocking the first NCCL collective in the chunk manager.
    if trace.nccl_gather_s and trace.nccl_reduce_s and trace.world == world_size:
        try:
            rank = int(dist.get_rank())
        except (RuntimeError, ValueError):
            rank = 0
        boot_result = getattr(wrapped, "search_result", None)
        if boot_result is not None:
            rank0_plan_box = (
                [boot_result.cfg, boot_result.block_map] if rank == 0 else [None, None]
            )
            bcast_ok = False
            try:
                dist.broadcast_object_list(rank0_plan_box, src=0)
                bcast_ok = True
            except (RuntimeError, ValueError) as exc:
                LOG.warning(
                    "ProTrain: warm-cache plan consistency broadcast failed "
                    "(%s); cannot verify cross-rank plan agreement.",
                    exc,
                )
            if bcast_ok and rank != 0:
                rank0_cfg, rank0_block_map = (
                    rank0_plan_box[0],
                    rank0_plan_box[1],
                )
                if (
                    rank0_cfg != boot_result.cfg
                    or rank0_block_map != boot_result.block_map
                ):
                    raise RuntimeError(
                        "ProTrain invariant violated: bootstrap search "
                        "converged to different plans across ranks. "
                        f"rank={rank} got cfg={boot_result.cfg}, "
                        f"block_map_len={len(boot_result.block_map)}; "
                        f"rank0 got cfg={rank0_cfg}, "
                        f"block_map_len={len(rank0_block_map)}. "
                        "On mixed-SKU rigs this is usually per-rank "
                        "measure_compute_rate outliers flipping the cost "
                        "model's 1%-noise-band tie-breaker. PR #20 "
                        "broadcasts gpu_compute_tflops to prevent this — "
                        "if you see this error, the broadcast did not "
                        "execute (rank-0 outlier? skipped path?)."
                    )
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
        rank = int(dist.get_rank())
    except (RuntimeError, ValueError):
        rank = 0

    local_ok = True
    gather_table = None
    reduce_table = None
    try:
        gather_table, reduce_table = measure_nccl(world_size)
    except (RuntimeError, ImportError) as exc:
        LOG.warning(
            "ProTrain: NCCL re-measurement failed on rank %d (%s); will "
            "coordinate bail with other ranks to avoid divergent plans.",
            rank,
            exc,
        )
        local_ok = False

    # Coordinate per-rank measurement success: if ANY rank failed, ALL ranks
    # bail. Otherwise rank-0 tables are broadcast so every rank feeds search()
    # identical inputs (tie-breaks in the cost model are bandwidth-sensitive,
    # so unsynchronized tables can produce divergent best_cfg).
    status_box = [1 if local_ok else 0]
    try:
        dist.broadcast_object_list(status_box, src=0)
    except (RuntimeError, ValueError) as exc:
        LOG.warning(
            "ProTrain: NCCL re-measurement status broadcast failed (%s); "
            "leaving trace with empty tables to avoid plan divergence.",
            exc,
        )
        return (False, False)
    rank0_ok = bool(status_box[0])
    if not rank0_ok or not local_ok:
        LOG.warning(
            "ProTrain: NCCL re-measurement bail — rank0_ok=%s, local_ok=%s; "
            "leaving trace with empty tables consistently across ranks.",
            rank0_ok,
            local_ok,
        )
        return (False, False)

    # Broadcast rank-0's measured tables so every rank's search() sees
    # identical inputs. Wrap the two tables in a list so a single
    # broadcast_object_list call shares them.
    table_box = [gather_table, reduce_table] if rank == 0 else [None, None]
    try:
        dist.broadcast_object_list(table_box, src=0)
    except (RuntimeError, ValueError) as exc:
        LOG.warning(
            "ProTrain: NCCL table broadcast failed (%s); leaving trace "
            "with empty tables to avoid plan divergence.",
            exc,
        )
        return (False, False)
    gather_table, reduce_table = table_box[0], table_box[1]

    new_trace = dataclasses.replace(
        trace,
        nccl_gather_s=gather_table,
        nccl_reduce_s=reduce_table,
        world=world_size,
    )

    # Save under live-world key so future multi-rank runs hit the cache.
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

    # Reuse hw and CPU budget unchanged.
    cpu_capacity = getattr(wrapped, "_cpu_capacity_bytes", None)
    new_result = search(
        new_trace, layout, capacity, hw, cpu_capacity_bytes=cpu_capacity
    )

    # Even with broadcast inputs, defensively assert that every rank
    # converged on the same (cfg, block_map). A divergence here would mean
    # cost-model tie-break non-determinism that we MUST surface, not paper
    # over: training under different plans corrupts gradient sync.
    rank0_plan_box = (
        [new_result.cfg, new_result.block_map] if rank == 0 else [None, None]
    )
    try:
        dist.broadcast_object_list(rank0_plan_box, src=0)
    except (RuntimeError, ValueError) as exc:
        LOG.warning(
            "ProTrain: post-search plan broadcast failed (%s); cannot "
            "verify cross-rank plan consistency, bailing.",
            exc,
        )
        return (False, False)
    if rank != 0:
        rank0_cfg, rank0_block_map = rank0_plan_box[0], rank0_plan_box[1]
        if rank0_cfg != new_result.cfg or rank0_block_map != new_result.block_map:
            raise RuntimeError(
                "ProTrain invariant violated: post-NCCL search converged "
                f"to different plans across ranks. rank={rank} got "
                f"cfg={new_result.cfg}, block_map={new_result.block_map}; "
                f"rank0 got cfg={rank0_cfg}, block_map={rank0_block_map}. "
                "This indicates non-determinism in the cost model's "
                "tie-break logic — file a bug."
            )

    cfg_changed = (
        new_result.cfg != wrapped.search_result.cfg
        or new_result.block_map != wrapped.search_result.block_map
    )
    if cfg_changed:
        # Fail-fast: bootstrap runtime can't rebuild mid-flight; stash telemetry then raise.
        LOG.warning(
            "ProTrain: post-NCCL search picked a different config than "
            "the bootstrap prediction. cfg %s -> %s; stashing the "
            "post-NCCL plan on WrappedModel.post_nccl_search_result for "
            "telemetry. Reaching this branch suggests early dist init "
            "was skipped — check cfg.ddp_backend / launcher env.",
            wrapped.search_result.cfg,
            new_result.cfg,
        )
        # Stash telemetry before raising so callers can introspect both plans.
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
        # Same cfg/block_map: publish refreshed numbers onto live fields.
        wrapped.search_result = new_result
        wrapped._trace = new_trace  # type: ignore[attr-defined]

    return (True, cfg_changed)


_LORA_FACTOR_NAME_MARKERS: tuple[str, ...] = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


def _detect_nvlink_topology() -> bool:
    """Return True when every visible GPU pair has at least one active NVLink.

    Path B's coalesced grad sync is a clear win on PCIe-class consumer rigs
    (-68% NCCL collective count → +15% sps/rank on 3090 PCIe 4-rank), but on
    NV-class fabric (300+ GB/s) the native bucketed allreduce is fast enough
    that Path B's serialization on the broadcasting rank becomes net overhead
    (measured -55% sps/rank on 2× A100-SXM4-80GB NVLink). When the user
    leaves ``protrain_own_lora_grad_sync`` at its ``None`` default, this
    detector picks the topology-appropriate behavior.

    Detection is conservative: single-GPU returns False (default-True
    semantics carry through unchanged); any nvidia-smi failure returns False
    (preserves the pre-topology-aware default of enabling Path B); a single
    pair without an active NVLink returns False (any heterogeneous topology
    is treated as PCIe-class for safety).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        n_visible = torch.cuda.device_count()
        if n_visible < 2:
            return False
    except (ImportError, RuntimeError):
        return False

    try:
        import subprocess

        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    gpu_lines = [ln for ln in proc.stdout.splitlines() if ln.lstrip().startswith("GPU")]
    gpu_lines = [ln for ln in gpu_lines if "\t" in ln or "  " in ln]
    if len(gpu_lines) < n_visible:
        return False

    for ln in gpu_lines[:n_visible]:
        cells = ln.split()
        if len(cells) < 1 + n_visible:
            return False
        pair_cells = cells[1 : 1 + n_visible]
        for cell in pair_cells:
            if cell == "X":
                continue
            if not cell.startswith("NV"):
                return False
    return True


def _resolve_path_b_default(cfg) -> tuple[bool, str]:
    """Return ``(enabled, reason)`` for ``protrain_own_lora_grad_sync``.

    Honors explicit True/False set by the user. Resolves ``None`` against the
    detected GPU topology: NVLink → False, otherwise → True.
    """
    explicit = getattr(cfg, "protrain_own_lora_grad_sync", None)
    if explicit is True:
        return True, "explicit cfg.protrain_own_lora_grad_sync=True"
    if explicit is False:
        return False, "explicit cfg.protrain_own_lora_grad_sync=False"
    is_nvlink = _detect_nvlink_topology()
    if is_nvlink:
        return False, "auto (NVLink topology detected; native NCCL is faster)"
    return True, "auto (non-NVLink topology; coalesced sync amortizes launch tax)"


def _discover_lora_params(model) -> tuple[list[str], list]:
    """Return (fully-qualified names, params) for all trainable PEFT-LoRA factors.

    PEFT injects LoRA factor modules with attribute names ``lora_A``,
    ``lora_B``, ``lora_embedding_A``, ``lora_embedding_B``, and
    ``lora_magnitude_vector`` (DoRA). Each contains an ``nn.ModuleDict``
    keyed by adapter name, so the fully-qualified parameter name typically
    reads ``base_model.model....lora_A.default.weight``.

    The match is on a sentinel substring (``.lora_A.`` etc.) of
    ``f".{name}."`` to handle both leading-dot and trailing-dot cases.
    Only ``requires_grad=True`` params are returned.
    """
    names: list[str] = []
    params: list = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        probe = f".{name}."
        for marker in _LORA_FACTOR_NAME_MARKERS:
            if marker in probe:
                names.append(name)
                params.append(p)
                break
    return names, params


def _register_lora_ddp_ignore(model, lora_names: list[str]) -> None:
    """Union LoRA names into ``model._ddp_params_and_buffers_to_ignore``.

    Mirrors the snapshot pattern already used in
    ``ChunkManager.materialize_offload``: pre-protrain values are stashed
    under ``_protrain_ddp_original_ignore`` so teardown can restore them.
    A pre-existing snapshot (set by chunk_manager) is preserved as-is — the
    chunk-managed names + LoRA names union into the live attribute.
    """
    existing = getattr(model, "_ddp_params_and_buffers_to_ignore", None)
    if not hasattr(model, "_protrain_ddp_original_ignore"):
        model._protrain_ddp_original_ignore = (  # type: ignore[attr-defined]
            None if existing is None else list(existing)
        )
    merged = set(existing or []) | set(lora_names)
    model._ddp_params_and_buffers_to_ignore = list(merged)  # type: ignore[attr-defined]


def _maybe_bypass_ddp_for_mode_c(trainer, wrapped) -> bool:
    """Force ``DistributedType.NO`` when Mode C is active on a multi-rank launch.

    Accelerate's default multi-GPU path wraps the model in DDP at ``prepare()``.
    DDP's bucketed all-reduce double-syncs gradients on top of ProTrain's
    per-chunk ``reduce_scatter`` (sharded chunks) and ``all_reduce`` (persistent
    chunks), corrupting the effective update. ProTrain's per-chunk collectives
    are issued directly from ``ChunkManager.reduce_grads_and_offload`` and
    parameter-level ``register_post_accumulate_grad_hook`` callbacks — both
    independent of DDP — so bypassing the DDP wrap leaves the cross-rank grad
    sync intact via ProTrain's own path. Mirrors the pattern used by
    ``DistributedParallelMixin.create_accelerator_and_postprocess`` for the
    Context Parallel + non-FSDP case (see
    ``core/trainers/mixins/distributed_parallel.py``).

    Returns True iff the override fired.
    """
    chunk_manager = getattr(wrapped, "chunk_manager", None)
    if chunk_manager is None or not bool(getattr(chunk_manager, "zero3_shard", False)):
        return False

    try:
        import torch.distributed as dist
    except ImportError:
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if int(dist.get_world_size()) <= 1:
        return False

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        return False

    try:
        from accelerate import PartialState
        from accelerate.utils import DistributedType
    except ImportError:
        return False

    prior = getattr(accelerator.state, "distributed_type", None)
    if prior == DistributedType.NO:
        return False

    accelerator.state.distributed_type = DistributedType.NO
    accelerator.state._shared_state["distributed_type"] = DistributedType.NO
    PartialState().distributed_type = DistributedType.NO

    LOG.warning(
        "ProTrain Mode C bypass: forcing accelerator.state.distributed_type "
        "from %s -> DistributedType.NO so Accelerate.prepare() skips the "
        "DDP wrap. ProTrain owns cross-rank grad sync via per-chunk "
        "reduce_scatter (sharded) / all_reduce (persistent); DDP would "
        "double-sync. Cross-rank loss aggregation in trainer logs may show "
        "per-rank values instead of mean — this is expected.",
        prior,
    )
    return True


def _maybe_bypass_ddp_for_path_b(trainer) -> bool:
    """Force ``DistributedType.NO`` when Path B owns sync for all trainable params.

    When ``protrain_own_lora_grad_sync`` is True and every trainable param
    is a LoRA factor whose name now lives in
    ``_ddp_params_and_buffers_to_ignore``, DDP's own ``__init__`` would
    raise ``RuntimeError: DistributedDataParallel is not needed when a
    module doesn't have any parameter that requires a gradient``. Mirrors
    the Mode C bypass — sets accelerator.state.distributed_type to NO so
    Accelerate.prepare() skips the DDP wrap entirely. ProTrain's
    ``_sync_lora_grads_path_b`` is the cross-rank sync.

    Returns True iff the override fired.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if int(dist.get_world_size()) <= 1:
        return False

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        return False

    try:
        from accelerate import PartialState
        from accelerate.utils import DistributedType
    except ImportError:
        return False

    prior = getattr(accelerator.state, "distributed_type", None)
    if prior == DistributedType.NO:
        return False

    accelerator.state.distributed_type = DistributedType.NO
    accelerator.state._shared_state["distributed_type"] = DistributedType.NO
    PartialState().distributed_type = DistributedType.NO

    LOG.warning(
        "ProTrain Path B bypass: all trainable params are LoRA factors "
        "marked in _ddp_params_and_buffers_to_ignore, so DDP would refuse "
        "to wrap. Forcing accelerator.state.distributed_type from %s -> "
        "DistributedType.NO; ProTrain owns LoRA grad-sync via flattened "
        "all_reduce in _ProTrainOptimizer.step(). Cross-rank loss "
        "aggregation in trainer logs may show per-rank values instead of "
        "mean — this is expected.",
        prior,
    )
    return True


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

    # Snapshot hyperparams now; Accelerate.prepare may wrap the optimizer later.
    args = trainer.args
    rebuild_lr = float(args.learning_rate)
    rebuild_betas = (float(args.adam_beta1), float(args.adam_beta2))
    rebuild_eps = float(args.adam_epsilon)
    rebuild_weight_decay = float(args.weight_decay)
    rebuild_optimizer_name = _resolve_optimizer_name(args, cfg)
    rebuild_huge_threshold = int(
        getattr(cfg, "protrain_persistent_huge_param_threshold_bytes", None)
        or 512 * 1024 * 1024
    )

    def _patched(resume_from_checkpoint, model=None) -> None:
        # Resolve chunk_manager lazily through wrapped so reorders can't strand the closure.
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        if chunk_manager is None:
            LOG.debug(
                "ProTrain resume hook: wrapped.chunk_manager is None; "
                "delegating to the original _load_from_checkpoint."
            )
            return original_load(resume_from_checkpoint, model)

        # Gate covers both replicated (_cpu_slots) and sharded (_chunk_shards) offload.
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
        # Drop GPU adapter so the rebuild can reconstruct against fresh storage.
        chunk_manager.gpu_optim = None

        # restore_to_gpu rebinds every param.data to standalone full-shape GPU storage.
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

        # HF's _load_from_checkpoint signature varies; forward model only when provided.
        if model is None:
            original_load(resume_from_checkpoint)
        else:
            original_load(resume_from_checkpoint, model)

        # Re-materialize offload from freshly-loaded weights.
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

        # Rebuild optimizer adapters against the fresh shard_param views.
        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        # Path B: re-discover LoRA params on the freshly-loaded model so the
        # rebuilt optimizer's _lora_owned_params references match the
        # post-restore param storage (restore_to_gpu rebinds .data).
        rebuild_lora_params: list | None = None
        _path_b_active, _ = _resolve_path_b_default(cfg)
        if _path_b_active:
            try:
                import torch.distributed as _dist

                if (
                    _dist.is_available()
                    and _dist.is_initialized()
                    and int(_dist.get_world_size()) > 1
                ):
                    _, rebuild_lora_params = _discover_lora_params(trainer.model)
            except (ImportError, RuntimeError, ValueError):
                rebuild_lora_params = None

        try:
            new_optim = protrain_optimizer_wrapper(
                wrapped,
                lr=rebuild_lr,
                betas=rebuild_betas,
                eps=rebuild_eps,
                weight_decay=rebuild_weight_decay,
                optimizer_name=rebuild_optimizer_name,
                huge_param_threshold_bytes=rebuild_huge_threshold,
                lora_owned_params=rebuild_lora_params,
            )
        except Exception:
            LOG.exception(
                "ProTrain resume hook: protrain_optimizer_wrapper rebuild "
                "failed after materialize_offload; runtime can't continue "
                "without an optimizer. Re-raising."
            )
            raise

        # Safe swap: Accelerate.prepare runs downstream of the load call site.
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
    """Return True iff plugin is registered and protrain_auto_memory is on."""
    if not getattr(cfg, "protrain_auto_memory", False):
        return False
    plugins = getattr(cfg, "plugins", None) or []
    return _has_protrain_plugin(plugins)


def _build_hardware_profile(cfg):
    """Construct a HardwareProfile with cfg-driven zero3_shard auto-detect."""
    from axolotl.integrations.protrain.api.hardware import _resolve_world_size

    world_size = _resolve_world_size()

    # Mirror protrain_model_wrapper's zero3_shard auto-detect.
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
    """Plugin for ProTrain integration with Axolotl."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.protrain.args.ProTrainArgs"

    def get_training_args(self, cfg):
        """Gate ``save_only_model`` on whether ProTrain owns the optim shard."""
        if not _is_plugin_active(cfg):
            return None
        save_optim_state = bool(getattr(cfg, "protrain_save_optimizer_state", False))
        return {"save_only_model": not save_optim_state}

    def pre_model_load(self, cfg) -> None:
        """Fail fast if peft / transformers API surface has drifted."""
        # Loud-when-inert: fire BEFORE the active gate so plugin-listed-but-disabled runs surface.
        _maybe_warn_inert_plugin(cfg)
        if not _is_plugin_active(cfg):
            return
        from axolotl.integrations.protrain.check import (
            assert_supported_peft_transformers_surface,
            warn_on_unvalidated_versions,
        )

        assert_supported_peft_transformers_surface()
        warn_on_unvalidated_versions()

        # Propagate the YAML knob into the cost-model module default before any
        # searcher / wrapper call site consumes _compute_ckpt_chain_bytes.
        from axolotl.integrations.protrain.cost.memory import (
            set_default_ckpt_internal_residual_factor,
        )

        residual_factor = getattr(cfg, "protrain_ckpt_internal_residual_factor", None)
        if residual_factor is not None:
            set_default_ckpt_internal_residual_factor(float(residual_factor))

    def post_model_load(self, cfg, model: "nn.Module") -> None:
        """Wrap the post-adapter model with the ProTrain runtime."""
        if not _is_plugin_active(cfg):
            return

        # Idempotency: same-model re-entry no-ops; different-model clears stale wrapper.
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
            # Deterministic teardown so pinned pools/hooks release before the next wrap.
            try:
                existing.close()
            except Exception as exc:  # noqa: BLE001
                # Fail closed; swallowing would leak pinned pools / hooks / Adam worker.
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

        # Bring up dist.init before HW profile so it reports true world size.
        _early_init_dist_for_nccl(cfg)

        # Target device is a hint; wrapper owns placement. No model.to() here.
        import os as _os

        target_device = None
        try:
            import torch as _torch
        except ImportError:
            _torch = None  # type: ignore[assignment]
        if _torch is not None and _torch.cuda.is_available():
            # Skip on hf_device_map loads: device map already pins shards.
            hf_device_map = getattr(model, "hf_device_map", None)
            if hf_device_map:
                LOG.info(
                    "ProTrain: model has hf_device_map=%s; deferring "
                    "device selection to the wrapper (target_device=None).",
                    hf_device_map,
                )
            else:
                # Defensive parse for non-numeric / out-of-range LOCAL_RANK.
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
                    # Metadata hint for plugin-less callers; kwarg path is canonical.
                    try:
                        model._protrain_target_device = target_device  # type: ignore[attr-defined]
                    except (AttributeError, TypeError):
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

        # Pull knobs / overrides off the merged cfg.
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

        # auto_mode default True; wrapper picks (force_persist, zero3) post-search.
        auto_mode = getattr(cfg, "protrain_auto_mode", True)
        if auto_mode is None:
            auto_mode = True

        # Mode B parity knob: force replicated CPU-offload (force_all_persistent=False,
        # zero3_shard=False) when auto_mode is off. The args validator already rejects
        # multiple force_* flags being set, so this branch is only reachable when no
        # other force flag is true. No-op under auto_mode (consistent with how
        # force_all_persistent / zero3_shard behave there).
        force_replicated_cpu_offload = bool(
            getattr(cfg, "protrain_force_replicated_cpu_offload", False)
        )
        if force_replicated_cpu_offload and not auto_mode:
            force_all_persistent = False
            zero3_shard = False
            LOG.info(
                "ProTrain: protrain_force_replicated_cpu_offload=True with "
                "auto_mode=False; forcing Mode B "
                "(force_all_persistent=False, zero3_shard=False)."
            )

        # Fused LoRA MLP backward kernel + offloaded-activation chunk placeholders
        # crash with LoRA_MLPBackward shape-mismatch (v61); pin n_offload=0.
        forbid_activation_offload = bool(getattr(cfg, "lora_mlp_kernel", False))
        if forbid_activation_offload:
            LOG.info(
                "ProTrain: cfg.lora_mlp_kernel=True; searcher will refuse "
                "n_offload>0 candidates."
            )

        # PR #17c: defensive searcher tie-break on non-NVLink multi-rank rigs.
        # Default True; set False once PR #17b lands the chunk re-gather fix.
        prefer_no_offload_on_non_nvlink_raw = getattr(
            cfg, "protrain_prefer_no_offload_on_non_nvlink", True
        )
        prefer_no_offload_on_non_nvlink = (
            True
            if prefer_no_offload_on_non_nvlink_raw is None
            else bool(prefer_no_offload_on_non_nvlink_raw)
        )

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
            forbid_activation_offload=forbid_activation_offload,
            prefer_no_offload_on_non_nvlink=prefer_no_offload_on_non_nvlink,
        )

        cfg._protrain_wrapped = wrapped  # type: ignore[attr-defined]

        picked = wrapped.search_result.cfg
        # Read effective mode from chunk_manager since auto_mode may have overridden user flags.
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
            # post_model_load was skipped; fall through to default optimizer.
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

        huge_threshold = int(
            getattr(cfg, "protrain_persistent_huge_param_threshold_bytes", None)
            or 512 * 1024 * 1024
        )
        return protrain_optimizer_wrapper(
            wrapped,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            huge_param_threshold_bytes=huge_threshold,
        )

    def post_trainer_create(self, cfg, trainer: "Trainer") -> None:
        """Install the ProTrain optimizer on the trainer + DDP-composition detection.

        Why ``post_trainer_create`` and not ``post_model_load``: the trainer's
        ``optim_cls_and_kwargs`` and dataloader configuration are only available
        after the HF ``Trainer`` is constructed. The persistent-chunk optimizer
        needs the trainer's optimizer name (to route adamw_torch / adamw_8bit /
        paged_adamw_8bit / adamw_apex_fused), and the DDP-composition detection
        needs the resolved ``accelerator`` from the trainer to decide whether to
        engage Mode A / B / C. Installing earlier (in ``post_model_load``) would
        force ProTrain to either duplicate trainer state-resolution logic or
        miss the optimizer-name routing. See ``DESIGN.md`` §6 for the full
        sequencing argument.
        """
        if not _is_plugin_active(cfg):
            return

        # Idempotency: re-entrant calls would stack duplicate hooks/callbacks.
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

        # Unlock Mode C on multi-rank non-NVLink rigs: must run before the
        # _inner_training_loop call to accelerator.prepare(self.model) wraps DDP.
        ddp_bypassed = _maybe_bypass_ddp_for_mode_c(trainer, wrapped)

        # Path B: ProTrain-owned LoRA grad sync. Discovery + DDP-ignore
        # registration must fire BEFORE Accelerate.prepare wraps DDP (which reads
        # _ddp_params_and_buffers_to_ignore at construction time). Restricted to
        # the DDP-active path: under Mode C bypass, chunk_manager already owns
        # the per-chunk sync and engaging Path B would risk double-syncing the
        # persistent-chunk LoRA grads.
        path_b_active = False
        lora_owned_params: list = []
        path_b_enabled, path_b_reason = _resolve_path_b_default(cfg)
        LOG.info(
            "ProTrain: protrain_own_lora_grad_sync resolved to %s (%s)",
            path_b_enabled,
            path_b_reason,
        )
        if path_b_enabled:
            if ddp_bypassed:
                LOG.info(
                    "ProTrain: protrain_own_lora_grad_sync=True but Mode C "
                    "bypass fired (DDP is bypassed); Path B is a no-op here. "
                    "Chunk_manager's per-chunk reduce_scatter / all_reduce "
                    "already owns cross-rank sync for sharded chunks."
                )
            else:
                lora_names, lora_owned_params = _discover_lora_params(trainer.model)
                # Multi-rank guard at the registration site keeps single-rank
                # runs from accumulating dead state on the attribute.
                try:
                    import torch.distributed as _dist

                    _is_multi_rank = (
                        _dist.is_available()
                        and _dist.is_initialized()
                        and int(_dist.get_world_size()) > 1
                    )
                except (ImportError, RuntimeError, ValueError):
                    _is_multi_rank = False
                if lora_names and _is_multi_rank:
                    _register_lora_ddp_ignore(trainer.model, lora_names)
                    path_b_active = True
                    # If ALL trainable params are LoRA factors (typical for
                    # LoRA/QLoRA fine-tunes), DDP would be left with no
                    # gradient-bearing params after the ignore filter and
                    # would raise "DistributedDataParallel is not needed
                    # when a module doesn't have any parameter that
                    # requires a gradient." Detect this and force
                    # DistributedType.NO so Accelerate.prepare() skips the
                    # DDP wrap entirely — Path B owns sync at the optimizer.
                    n_trainable = sum(
                        1 for p in trainer.model.parameters() if p.requires_grad
                    )
                    if n_trainable == len(lora_names):
                        _maybe_bypass_ddp_for_path_b(trainer)
                        # Track the bypass for downstream DDP-composition checks.
                        ddp_bypassed = True
                    LOG.info(
                        "ProTrain Path B: registered %d trainable LoRA "
                        "param names in model._ddp_params_and_buffers_to_ignore "
                        "(n_trainable=%d); ProTrain will own grad-sync via "
                        "flattened all_reduce in _ProTrainOptimizer.step().",
                        len(lora_names),
                        n_trainable,
                    )
                elif lora_names and not _is_multi_rank:
                    LOG.info(
                        "ProTrain Path B: %d LoRA params found but world_size "
                        "<= 1; flag is a no-op (single-rank).",
                        len(lora_names),
                    )
                    # Pass refs through anyway so the optimizer short-circuits
                    # on world<=1 rather than carrying stale state.
                else:
                    LOG.info(
                        "ProTrain Path B: protrain_own_lora_grad_sync=True "
                        "but no trainable LoRA factor params found; flag is "
                        "a no-op."
                    )

        from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

        args = trainer.args
        optimizer_name = getattr(args, "optim", None) or getattr(cfg, "optimizer", None)
        if optimizer_name is not None and not isinstance(optimizer_name, str):
            optimizer_name = getattr(optimizer_name, "value", str(optimizer_name))
        huge_threshold = int(
            getattr(cfg, "protrain_persistent_huge_param_threshold_bytes", None)
            or 512 * 1024 * 1024
        )
        optim = protrain_optimizer_wrapper(
            wrapped,
            lr=float(args.learning_rate),
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_epsilon),
            weight_decay=float(args.weight_decay),
            optimizer_name=optimizer_name,
            huge_param_threshold_bytes=huge_threshold,
            lora_owned_params=lora_owned_params if path_b_active else None,
        )

        # state_dict/load_state_dict empty-shell behavior lives in _ProTrainOptimizer.
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

        # Optimizer-state checkpoint/resume; opt-in via protrain_save_optimizer_state.
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

        # DDP composition detection: defer grad reduce to DDP when wrapped.
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
        # Diagnostic: state at post_trainer_create entry for v53 OOM investigation.
        try:
            _diag_inner = trainer.model
            if isinstance(_diag_inner, DistributedDataParallel):
                _diag_inner = _diag_inner.module
            _diag_has_attr = hasattr(_diag_inner, "_ddp_params_and_buffers_to_ignore")
            _diag_ignore_size = (
                len(_diag_inner._ddp_params_and_buffers_to_ignore)  # type: ignore[attr-defined]
                if _diag_has_attr
                else 0
            )
            _diag_alloc_gib = (
                torch.cuda.memory_allocated() / (1 << 30)
                if torch.cuda.is_available()
                else 0.0
            )
            LOG.warning(
                "[protrain-diag] post_trainer_create entry: "
                "alloc=%.2f GiB is_ddp_wrapped=%s "
                "inner_has_ignore_attr=%s inner_ignore_size=%d",
                _diag_alloc_gib,
                is_ddp,
                _diag_has_attr,
                _diag_ignore_size,
            )
        except Exception as _diag_exc:  # noqa: BLE001
            LOG.warning(
                "[protrain-diag] post_trainer_create logging failed: %s",
                _diag_exc,
            )
        if is_ddp:
            # Fallback safety net: bypass should have prevented this.
            chunk_manager = cast("ChunkManager", wrapped.chunk_manager)
            if getattr(chunk_manager, "zero3_shard", False):
                raise RuntimeError(
                    "ProTrain: DDP wrapping detected with active "
                    "zero3_shard=True even though the Mode C bypass attempted "
                    f"to set distributed_type=NO (ddp_bypassed={ddp_bypassed}). "
                    "Non-persistent sharded chunks call reduce_scatter via "
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
            # Mode C bypass intentionally puts us here; demote to info to avoid noise.
            log_fn = LOG.info if ddp_bypassed else LOG.warning
            log_fn(
                "ProTrain: multi-rank init (world_size=%d) detected with "
                "trainer.model NOT wrapped in DistributedDataParallel "
                "(ddp_bypassed=%s); ProTrain's internal per-chunk grad "
                "reduce_scatter / all_reduce path is the cross-rank sync.",
                torch.distributed.get_world_size(),
                ddp_bypassed,
            )

        # Re-measure NCCL now that dist is up; no-op if tables already populated.
        _remeasure_nccl_and_research(wrapped)

        # Eager per-chunk NCCL warmup: pay ncclCommInitRank cost here, not
        # during the first training iter's autograd-internal dispatch.
        if bool(getattr(cfg, "protrain_eager_nccl_warmup", True)):
            chunk_manager_for_warmup = getattr(wrapped, "chunk_manager", None)
            if chunk_manager_for_warmup is not None:
                accelerator = getattr(trainer, "accelerator", None)
                warmup_device = None
                if accelerator is not None:
                    warmup_device = getattr(accelerator, "device", None)
                if warmup_device is None:
                    warmup_device = getattr(chunk_manager_for_warmup, "device", None)
                if warmup_device is not None:
                    _eager_nccl_warmup(chunk_manager_for_warmup, warmup_device)
                else:
                    LOG.warning(
                        "ProTrain: skipping eager NCCL warm-up — could not "
                        "resolve a target device from trainer.accelerator or "
                        "chunk_manager. First iter will pay lazy "
                        "ncclCommInitRank cost as before."
                    )

        trainer._protrain_post_trainer_create_done = True  # type: ignore[attr-defined]


__all__ = ["ProTrainPlugin"]
