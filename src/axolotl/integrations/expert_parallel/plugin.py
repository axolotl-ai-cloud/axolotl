# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-Parallel (DeepEP) plugin for axolotl."""

from __future__ import annotations

from importlib.util import find_spec

import torch.distributed as dist

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ExpertParallelPlugin(BasePlugin):
    """Plugin that swaps MoE dispatch/combine for DeepEP-fused kernels.

    See DEEPEP_SETUP.md (install) and BENCHMARK.md (perf) at repo root.
    """

    def get_input_args(self):
        return "axolotl.integrations.expert_parallel.ExpertParallelArgs"

    def pre_model_load(self, cfg):
        if not self._is_ep_enabled(cfg):
            return

        if not self._deep_ep_available(cfg):
            return  # already-warned fallback path

        # Cross-cfg validation that args.py can't do (it only sees its own fields).
        self._validate_mesh_axes(cfg)

        from .experts_fn import kernel_to_registered_name, register_all

        register_all()

        # Auto-compose with the user's chosen local kernel. The kernels integration
        # has already validated that `experts_implementation in {eager, scattermoe}`
        # and `use_scattermoe: true` is consistent — we read the result and upgrade.
        local_kernel = self._infer_local_kernel(cfg)
        composite = kernel_to_registered_name(local_kernel)
        previous = getattr(cfg, "experts_implementation", None)
        cfg.experts_implementation = composite
        LOG.info(
            f"expert_parallel: experts_implementation {previous!r} -> {composite!r} "
            f"(local kernel: {local_kernel!r})"
        )

    def post_model_build(self, cfg, model):
        if not self._is_ep_enabled(cfg):
            return
        if not self._deep_ep_available(cfg):
            return

        from .buffer import configure_buffer
        from .shard import shard_expert_weights

        ep_group = self._resolve_ep_group(cfg)
        sharded = shard_expert_weights(model, ep_group)

        if sharded == 0:
            LOG.warning(
                "expert_parallel_enabled=true but no Experts modules were detected "
                "for sharding (model uses a non-canonical layout, or single-rank). "
                "DeepEP dispatch/combine will run as a no-op."
            )

        configure_buffer(
            ep_group=ep_group,
            num_nvl_bytes=cfg.expert_parallel_num_nvl_bytes,
            num_rdma_bytes=cfg.expert_parallel_num_rdma_bytes,
        )

    def post_model_load(self, cfg, model):
        """Propagate DDP-ignored params to the outermost model wrapper.

        `post_model_build` set `_ddp_params_and_buffers_to_ignore` on the inner
        model. After PEFT wraps it (in `PeftModel`), DDP wraps `PeftModel`, but
        DDP looks for the attribute on the top-level module — which is now
        `PeftModel`, not our inner model. Mirror the list up.
        """
        if not self._is_ep_enabled(cfg):
            return

        # Find the inner module that has the attribute (shard set it on whatever
        # was the top-level model at post_model_build time).
        inner = getattr(model, "base_model", model)
        # base_model may itself be wrapped (e.g., LoraModel.model). Recurse.
        while not hasattr(inner, "_ddp_params_and_buffers_to_ignore"):
            next_inner = getattr(inner, "model", None) or getattr(
                inner, "base_model", None
            )
            if next_inner is None or next_inner is inner:
                break
            inner = next_inner

        ignore_list = getattr(inner, "_ddp_params_and_buffers_to_ignore", None)
        if not ignore_list:
            return

        # PEFT prefixes parameter names. Re-resolve the list against the wrapper
        # so DDP can match by name.
        resolved: list[str] = []
        wrapper_param_names = {n for n, _ in model.named_parameters()}
        wrapper_buffer_names = {n for n, _ in model.named_buffers()}
        all_names = wrapper_param_names | wrapper_buffer_names

        for short_name in ignore_list:
            # Match either an exact suffix or with PEFT's `base_model.model.` prefix.
            for full in all_names:
                if (
                    full == short_name
                    or full.endswith("." + short_name)
                    or full.endswith(short_name)
                ):
                    resolved.append(full)

        # De-dup while preserving order.
        seen = set()
        resolved = [n for n in resolved if not (n in seen or seen.add(n))]

        existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
        model._ddp_params_and_buffers_to_ignore = existing + resolved
        LOG.info(
            f"expert_parallel: propagated {len(resolved)} DDP-ignored param "
            f"name(s) onto outer wrapper {type(model).__name__}."
        )

    @staticmethod
    def _infer_local_kernel(cfg) -> str:
        """Decide which local-experts kernel runs under DeepEP dispatch.

        Source-of-truth toggles:
        - Custom MoE kernels (ScatterMoE, SonicMoE): `use_scattermoe` /
          `use_sonicmoe` (the master flags from `kernels/args.py`). Note that
          `experts_implementation: scattermoe` is set BY the kernels validator
          AS A CONSEQUENCE of `use_scattermoe: true` — checking it would be
          redundant and would also misfire if a user set the string without
          the master flag.
        - Standard transformers kernels: `experts_implementation` directly
          (`eager`, `grouped_mm`, `batched_mm`).

        SonicMoE is currently a Gemma4-only direct rebind on
        `Gemma4TextExperts.forward`, not registered against
        `ALL_EXPERTS_FUNCTIONS`. Once it migrates to register via the
        `EXPERTS_ONLY_BLOCK` constant pattern (kernels/constants.py:60-67),
        this branch can return `"sonicmoe"` and the plugin will compose for
        free using a `_sonicmoe_local` helper.
        """
        if getattr(cfg, "use_scattermoe", False):
            return "scattermoe"

        if getattr(cfg, "use_sonicmoe", False):
            LOG.warning(
                "expert_parallel + use_sonicmoe: SonicMoE is currently a Gemma4-only "
                "direct rebind on Gemma4TextExperts.forward, not registered against "
                "ALL_EXPERTS_FUNCTIONS. Falling back to grouped_mm for the local-experts "
                "kernel under DeepEP. Once SonicMoE migrates to the EXPERTS_ONLY_BLOCK "
                "registration pattern, this composition will work automatically."
            )
            return "grouped_mm"

        ei = getattr(cfg, "experts_implementation", None)
        if ei in ("grouped_mm", "batched_mm"):
            return "grouped_mm"
        if ei == "eager":
            return "eager"
        # default: upstream-shipped fast kernel
        return "grouped_mm"

    # Cached 2D DeviceMesh when EP composes with FSDP. Set by `_resolve_ep_group`.
    _device_mesh = None

    @staticmethod
    def _resolve_ep_group(cfg):
        """Return the EP ProcessGroup.

        Three cases, in order of precedence:

        1. EP composed with FSDP (dp_shard_size > 1 AND expert_parallel_size > 1):
           build a 2D `DeviceMesh` with axes ("ep", "dp_shard") shape
           (ep_size, dp_shard_size). Return `mesh["ep"].get_group()` — strided
           groups orthogonal to accelerate's contiguous dp_shard groups.

           Layout in C-order: rank R has coords (R // dp_shard_size, R % dp_shard_size).
           For world=4, dp_shard=2, ep=2:
               rank 0 -> (ep=0, dp=0)
               rank 1 -> (ep=0, dp=1)
               rank 2 -> (ep=1, dp=0)
               rank 3 -> (ep=1, dp=1)
               EP groups (vary ep, fix dp): {0,2}, {1,3} — strided.
               dp_shard groups (vary dp, fix ep): {0,1}, {2,3} — contiguous,
                                                   matches accelerate's default.

        2. EP-only (ep_size == world_size): use `dist.group.WORLD`.

        3. ep_size in {None, 1}: EP disabled, return `dist.group.WORLD` (caller
           checks `_is_ep_enabled` before invoking).
        """
        if not dist.is_available() or not dist.is_initialized():
            return None

        world_size = dist.get_world_size()
        ep_size = getattr(cfg, "expert_parallel_size", 1) or 1
        dp_shard_size = getattr(cfg, "dp_shard_size", None) or 1
        tp_size = getattr(cfg, "tensor_parallel_size", None) or 1
        cp_size = getattr(cfg, "context_parallel_size", None) or 1

        if ep_size <= 1:
            return dist.group.WORLD

        # Validate the world_size = product check.
        product = ep_size * dp_shard_size * tp_size * cp_size
        if product != world_size:
            raise ValueError(
                f"expert_parallel_size ({ep_size}) * dp_shard_size ({dp_shard_size}) "
                f"* tensor_parallel_size ({tp_size}) * context_parallel_size ({cp_size}) "
                f"= {product}, but world_size = {world_size}. The product must equal "
                f"the world size for orthogonal mesh axes to be valid."
            )

        if ep_size == world_size:
            return dist.group.WORLD

        # Case 1: EP + FSDP — build 2D orthogonal mesh.
        if dp_shard_size > 1:
            if tp_size > 1 or cp_size > 1:
                # 3D+ meshes (ep × dp_shard × tp × cp) — punt for v1 with a clear error.
                raise NotImplementedError(
                    "EP composition with TP/CP not yet supported. Got "
                    f"ep={ep_size}, dp_shard={dp_shard_size}, tp={tp_size}, cp={cp_size}. "
                    "v1 supports only EP-only or EP × dp_shard."
                )
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh(
                "cuda",
                (ep_size, dp_shard_size),
                mesh_dim_names=("ep", "dp_shard"),
            )
            ExpertParallelPlugin._device_mesh = mesh
            LOG.info(
                f"expert_parallel: built 2D mesh shape={tuple(mesh.shape)} "
                f"axes={mesh.mesh_dim_names}; ep group "
                f"members={dist.get_process_group_ranks(mesh['ep'].get_group())}"
            )
            return mesh["ep"].get_group()

        # ep_size > 1, ep_size < world_size, dp_shard_size == 1 — invalid because
        # the product check above would have already failed. Unreachable but safe.
        raise ValueError(
            f"expert_parallel_size ({ep_size}) < world_size ({world_size}) "
            "without dp_shard_size > 1 to fill the remaining axes is not supported. "
            "Set dp_shard_size such that ep × dp_shard == world_size, or set "
            "expert_parallel_size = world_size for pure EP."
        )

    @staticmethod
    def _is_ep_enabled(cfg) -> bool:
        """EP is enabled when expert_parallel_size > 1 (mirrors TP / DP UX)."""
        ep_size = getattr(cfg, "expert_parallel_size", 1) or 1
        return ep_size > 1

    @staticmethod
    def _validate_mesh_axes(cfg) -> None:
        """Sanity-check the mesh-axis sizes early, with a clear error.

        `_resolve_ep_group` re-validates at process-group construction time;
        this catches misconfigured YAMLs before model loading wastes minutes.
        """
        ep_size = getattr(cfg, "expert_parallel_size", 1) or 1
        if ep_size <= 1:
            return

        if not (dist.is_available() and dist.is_initialized()):
            return  # validated at process-group time
        world_size = dist.get_world_size()
        if world_size <= 1:
            return  # single-rank context; mesh shapes are meaningless
        dp_shard_size = getattr(cfg, "dp_shard_size", None) or 1
        tp_size = getattr(cfg, "tensor_parallel_size", None) or 1
        cp_size = getattr(cfg, "context_parallel_size", None) or 1

        product = ep_size * dp_shard_size * tp_size * cp_size
        if product != world_size:
            raise ValueError(
                f"expert_parallel: world_size ({world_size}) must equal "
                f"expert_parallel_size ({ep_size}) * dp_shard_size ({dp_shard_size}) "
                f"* tensor_parallel_size ({tp_size}) * context_parallel_size ({cp_size}) "
                f"= {product}."
            )

    @staticmethod
    def _deep_ep_available(cfg) -> bool:
        if find_spec("deep_ep") is not None:
            return True
        msg = (
            "expert_parallel_enabled=true but `deep_ep` is not importable. "
            "See DEEPEP_SETUP.md."
        )
        if cfg.expert_parallel_fallback_on_unsupported:
            LOG.warning(msg + " Falling back to standard experts implementation.")
            return False
        raise ImportError(msg)
