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

import torch
import torch.distributed as dist

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def expert_shard_axis(mesh_dim_names) -> str | None:
    """The non-``ep`` mesh axis the routed experts FSDP-shard on under EP composition, or ``None``.

    Prefers ``dp_shard`` (EP×dp_shard: experts shard on the data axis); falls back to ``cp`` (EP×cp,
    where the cp ranks of an ep-group hold the SAME experts since cp shards the sequence, not the
    experts, so FSDP-sharding them on cp keeps each rank from holding the full ep-group slice). Returns
    ``None`` for pure EP (no secondary axis) or when there is no ``ep`` axis to compose with — those
    paths don't pre-wrap the experts here.
    """
    names = tuple(mesh_dim_names or ())
    if "ep" not in names:
        return None
    if "dp_shard" in names:
        return "dp_shard"
    if "cp" in names:
        return "cp"
    return None


class ExpertParallelPlugin(BasePlugin):
    """Plugin that swaps MoE dispatch/combine for DeepEP-fused kernels."""

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

        # Upgrade the user's chosen local kernel to its DeepEP-wrapped variant.
        local_kernel = self._infer_local_kernel(cfg)
        composite = kernel_to_registered_name(local_kernel)
        previous = getattr(cfg, "experts_implementation", None)
        cfg.experts_implementation = composite
        LOG.debug(
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
        from .experts_fn import set_token_capacity

        set_token_capacity(getattr(cfg, "expert_parallel_token_capacity", None))
        # Pure-EP path: register the grad-scale hook now. FSDP+EP defers
        # registration to `fully_shard_experts` (after experts become DTensors).
        if (cfg.dp_shard_size or 1) <= 1:
            ep_size = cfg.expert_parallel_size or 1
            self._register_expert_grad_scale(model, ep_size)

    def post_model_load(self, cfg, model):
        """Propagate DDP-ignored params to the outermost model wrapper.

        `post_model_build` set `_ddp_params_and_buffers_to_ignore` on the inner
        model. After PEFT wraps it (in `PeftModel`), DDP wraps `PeftModel`, but
        DDP looks for the attribute on the top-level module — which is now
        `PeftModel`, not our inner model. Mirror the list up.
        """
        if not self._is_ep_enabled(cfg):
            return

        self._register_padding_dispatch_hook(model)

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
        LOG.debug(
            f"expert_parallel: propagated {len(resolved)} DDP-ignored param "
            f"name(s) onto outer wrapper {type(model).__name__}."
        )

    @staticmethod
    def _register_padding_dispatch_hook(model) -> None:
        """Feed the batch's real-token mask to the DeepEP dispatch so padding tokens are
        not routed (they'd otherwise pile onto one expert and break intranode dispatch).

        A model-level forward pre-hook reads the 2D ``attention_mask`` (1=real, 0=pad) and
        stashes a flattened ``[B*S]`` bool mask; ``_deep_ep_forward`` sentinels those rows.
        Under sample packing there is no 2D mask, but the multipack collator pads partial
        packs to ``seq_len`` — those identical pad embeddings still pile onto one expert and
        break DeepEP intranode dispatch — so fall back to ``input_ids != pad_token_id``."""
        from .experts_fn import set_valid_token_mask

        if getattr(model, "_ep_padding_hook", False):
            return

        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)

        def _pre_hook(_module, args, kwargs):
            am = kwargs.get("attention_mask")
            if am is None and args:
                am = next(
                    (a for a in args if torch.is_tensor(a) and a.dim() == 2), None
                )
            if am is not None and am.dim() == 2:
                set_valid_token_mask((am != 0).reshape(-1))
                return args, kwargs
            # Packing (no 2D mask): exclude pad rows so they don't overload one expert.
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = next(
                    (
                        a
                        for a in args
                        if torch.is_tensor(a)
                        and a.dim() == 2
                        and a.dtype in (torch.long, torch.int, torch.int32)
                    ),
                    None,
                )
            set_valid_token_mask(
                (ids != pad_id).reshape(-1)
                if (ids is not None and pad_id is not None)
                else None
            )
            return args, kwargs

        model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        model._ep_padding_hook = True

    @staticmethod
    def _infer_local_kernel(cfg) -> str:
        """Decide which local-experts kernel runs under DeepEP dispatch.

        `use_scattermoe` / `use_sonicmoe` are the master flags from
        `kernels/args.py` and take precedence; otherwise fall back to
        `experts_implementation` (`eager` / `grouped_mm` / `batched_mm`).
        """
        if getattr(cfg, "use_scattermoe", False):
            return "scattermoe"

        if getattr(cfg, "use_sonicmoe", False):
            return "sonicmoe"

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
    def _accelerate_mesh():
        """Return accelerate's device_mesh, force-creating the Accelerator if
        needed. The AcceleratorState singleton makes this idempotent w.r.t.
        the trainer's later `Accelerator()` call.
        """
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState

        try:
            state = AcceleratorState()
        except (RuntimeError, AttributeError, ValueError) as e:
            # ValueError occurs from unittest due to no trainer constructing Accelerator()
            LOG.debug(f"expert_parallel: AcceleratorState() not ready: {e}")
            return None
        mesh = getattr(state, "device_mesh", None)
        if mesh is not None:
            return mesh
        try:
            Accelerator()
        except (RuntimeError, ValueError) as e:
            LOG.debug(f"expert_parallel: Accelerator() force-init failed: {e}")
            return None
        return getattr(AcceleratorState(), "device_mesh", None)

    @staticmethod
    def _resolve_ep_group(cfg):
        """Return the EP ProcessGroup.

        For FSDP+EP, returns `accelerate_mesh["ep"].get_group()` — the same
        process group that accelerate's parallelism_config built. For pure EP
        (ep_size == world_size, no FSDP), returns `dist.group.WORLD`.
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

        # EP composed with FSDP (`dp_shard`) and/or context parallel (`cp`) on orthogonal mesh
        # axes — read the ep group from accelerate's mesh, or build one ourselves if accelerate
        # hasn't (e.g., topology unit tests that drive `_resolve_ep_group` directly). Experts shard
        # on `ep` (tokens move via all-to-all); the sequence shards on `cp` (DSA attention gathers
        # the compressed KV on that axis); non-expert weights shard on `dp_shard`. TP is still
        # unsupported in composition.
        if dp_shard_size > 1 or cp_size > 1:
            if tp_size > 1:
                raise NotImplementedError(
                    "EP × TP composition not yet supported. Got "
                    f"ep={ep_size}, dp_shard={dp_shard_size}, tp={tp_size}, cp={cp_size}. "
                    "Supported: EP, EP × dp_shard, EP × cp, EP × cp × dp_shard."
                )
            mesh = ExpertParallelPlugin._accelerate_mesh()
            if mesh is None or "ep" not in (mesh.mesh_dim_names or ()):
                from torch.distributed.device_mesh import init_device_mesh

                # Fallback mesh from the >1 axes (ep outermost). Orthogonality of the ep/cp/dp
                # groups is what matters; accelerate's mesh is preferred when present so the ep
                # group matches the one used for the experts' FSDP exclusion.
                axes = [("ep", ep_size)]
                if cp_size > 1:
                    axes.append(("cp", cp_size))
                if dp_shard_size > 1:
                    axes.append(("dp_shard", dp_shard_size))
                mesh = init_device_mesh(
                    "cuda" if torch.cuda.is_available() else "cpu",
                    tuple(s for _, s in axes),
                    mesh_dim_names=tuple(n for n, _ in axes),
                )
            ExpertParallelPlugin._device_mesh = mesh
            LOG.debug(
                f"expert_parallel: ep mesh shape={tuple(mesh.shape)} "
                f"axes={mesh.mesh_dim_names}; ep group "
                f"members={dist.get_process_group_ranks(mesh['ep'].get_group())}"
            )
            return mesh["ep"].get_group()

        # ep_size > 1, ep_size < world_size, no dp_shard/cp to fill the rest — invalid.
        raise ValueError(
            f"expert_parallel_size ({ep_size}) < world_size ({world_size}) "
            "without dp_shard_size/context_parallel_size > 1 to fill the remaining axes is not "
            "supported. Set dp_shard_size and/or context_parallel_size such that "
            "ep × cp × dp_shard == world_size, or set expert_parallel_size = world_size for pure EP."
        )

    @staticmethod
    def _resolve_cp_group(cfg):
        """Return the context-parallel ProcessGroup (the `cp` axis of the EP mesh), or None when
        ``context_parallel_size <= 1``. The DSA attention shards the sequence on this axis (gathering
        the compressed KV across it); experts shard on the orthogonal ``ep`` axis. Reads the mesh
        built by ``_resolve_ep_group`` / accelerate."""
        cp_size = getattr(cfg, "context_parallel_size", None) or 1
        if cp_size <= 1:
            return None
        mesh = (
            ExpertParallelPlugin._device_mesh or ExpertParallelPlugin._accelerate_mesh()
        )
        if mesh is not None and "cp" in (mesh.mesh_dim_names or ()):
            return mesh["cp"].get_group()
        return None

    @staticmethod
    def fully_shard_experts(model, dp_shard_mesh, fsdp2_kwargs):
        """Pre-wrap each Experts module with FSDP on the `dp_shard` axis.

        Called from the patched `fsdp2_prepare_model` BEFORE the outer auto-wrap
        so experts become FSDPModules and the auto-wrap walker skips them.
        Inherits the outer wrap's policy (mp, offload, reshard) so inner/outer
        collective dtypes line up; only `mesh` is overridden.
        """
        from torch.distributed.fsdp import fully_shard

        from .shard import (
            _detect_experts_modules,
            _is_param_wrapper,
            _real_experts_base,
        )

        kwargs = dict(fsdp2_kwargs)
        kwargs["mesh"] = dp_shard_mesh
        kwargs.pop("ignored_params", None)

        for _name, module in _detect_experts_modules(model):
            fully_shard(module, **kwargs)

        # `target_parameters` expert LoRA lives on the ParamWrapper chain wrapping the experts
        # module (which `_detect_experts_modules` skips). Left to the outer decoder-layer auto-wrap
        # it shards on the FULL ep×dp mesh — i.e. ACROSS the ep axis — corrupting the per-ep-rank
        # expert slice (grads averaged over ranks owning different experts; save reconstructs the
        # wrong shape). Wrap the OUTERMOST expert ParamWrapper as its own FSDP unit on dp_shard:
        # its forward IS the fused-LoRA fastpath, so FSDP unshards the adapter (incl. the nested
        # inner wrapper's, which is not a separate unit) to plain tensors right before the kernel
        # reads them — sharded on the same axis as the weights, but gathered during use.
        all_pws = [m for _n, m in model.named_modules() if _is_param_wrapper(m)]
        inner = {getattr(pw, "base_layer", None) for pw in all_pws}
        outer_expert_pws = [
            pw
            for pw in all_pws
            if pw not in inner
            and _real_experts_base(pw) is not None
            and getattr(_real_experts_base(pw), "num_local_experts", None) is not None
        ]
        for pw in outer_expert_pws:
            fully_shard(pw, **kwargs)

        LOG.debug(
            f"expert_parallel: pre-wrapped Experts modules + {len(outer_expert_pws)} expert "
            f"ParamWrapper(s) on dp_shard mesh (size={dp_shard_mesh.size()})."
        )

        root = dp_shard_mesh._get_root_mesh()
        ep_size = (
            root["ep"].size()
            if root is not None and "ep" in (root.mesh_dim_names or ())
            else 1
        )
        ExpertParallelPlugin._register_expert_grad_scale(model, ep_size)

    @staticmethod
    def _register_expert_grad_scale(model, ep_size: int) -> int:
        """Scale expert weight grads by `1/ep_size` so EP / FSDP / FSDP+EP
        produce the same effective gradient.
        """
        from .shard import _detect_experts_modules

        if ep_size <= 1:
            return 0
        scale = 1.0 / ep_size

        def _scale(p):
            if p.grad is not None:
                p.grad.mul_(scale)

        n_hooks = 0
        for _name, module in _detect_experts_modules(model):
            for p in module.parameters(recurse=True):
                # Only trainable params have a grad to scale — and a hook can only be
                # registered on a tensor that requires grad. Under LoRA the base expert
                # weights are frozen (only the adapters train), so skip them.
                if not p.requires_grad:
                    continue
                p.register_post_accumulate_grad_hook(_scale)
                n_hooks += 1
        LOG.debug(
            f"expert_parallel: registered {n_hooks} expert grad-scale hooks "
            f"(scale = 1/{ep_size})"
        )
        return n_hooks

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
            "See the integration README for install instructions."
        )
        if cfg.expert_parallel_fallback_on_unsupported:
            LOG.warning(msg + " Falling back to standard experts implementation.")
            return False
        raise ImportError(msg)
