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

    # Idempotency: tables already populated → no-op.
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

        return protrain_optimizer_wrapper(
            wrapped,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
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
        if is_ddp:
            # DDP + zero3_shard double-synchronizes grads; hard-raise so user reconfigures.
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

        # Re-measure NCCL now that dist is up; no-op if tables already populated.
        _remeasure_nccl_and_research(wrapped)

        trainer._protrain_post_trainer_create_done = True  # type: ignore[attr-defined]


__all__ = ["ProTrainPlugin"]
