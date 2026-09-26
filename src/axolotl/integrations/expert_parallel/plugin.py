# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-Parallel plugin for axolotl (DeepEP or torch all-to-all dispatch)."""

from __future__ import annotations

import os
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
    """Plugin that swaps MoE dispatch/combine for expert-parallel token dispatch."""

    def get_input_args(self):
        return "axolotl.integrations.expert_parallel.ExpertParallelArgs"

    def pre_model_load(self, cfg):
        if not self._is_ep_enabled(cfg):
            return

        backend = self._resolve_backend(cfg)
        if backend is None:
            return  # already-warned fallback path

        # Cross-cfg validation that args.py can't do (it only sees its own fields).
        self._validate_mesh_axes(cfg)

        from .experts_fn import kernel_to_registered_name, register_all

        register_all()

        # Upgrade the user's chosen local kernel to its EP-wrapped variant.
        local_kernel = self._infer_local_kernel(cfg)
        composite = kernel_to_registered_name(local_kernel, backend)
        previous = getattr(cfg, "experts_implementation", None)
        cfg.experts_implementation = composite
        LOG.info(
            f"expert_parallel: backend={backend!r}, experts_implementation "
            f"{previous!r} -> {composite!r} (local kernel: {local_kernel!r})"
        )

    def post_model_build(self, cfg, model):
        if not self._is_ep_enabled(cfg):
            return
        backend = self._resolve_backend(cfg)
        if backend is None:
            return

        from .shard import shard_expert_weights

        ep_group = self._resolve_ep_group(cfg)
        sharded = shard_expert_weights(model, ep_group)

        if sharded == 0:
            message = (
                "expert_parallel_size > 1 but no Experts modules were detected for "
                "sharding (the model does not use transformers' canonical 3-D "
                "gate_up_proj/down_proj experts layout)."
            )
            if ep_group is not None and dist.get_world_size(ep_group) > 1:
                raise ValueError(
                    message + " Expert parallelism cannot run on this model; "
                    "set expert_parallel_size: 1."
                )
            LOG.warning(message + " Expert-parallel dispatch/combine is a no-op.")

        chunks = getattr(cfg, "expert_parallel_dispatch_chunks", None) or 1
        if backend == "deep_ep":
            from .buffer import configure_buffer

            configure_buffer(
                ep_group=ep_group,
                num_nvl_bytes=cfg.expert_parallel_num_nvl_bytes,
                num_rdma_bytes=cfg.expert_parallel_num_rdma_bytes,
            )
            if chunks > 1:
                LOG.warning(
                    "expert_parallel_dispatch_chunks only applies to the torch backend; "
                    "ignored under deep_ep."
                )
        else:
            from .experts_fn import set_dispatch_chunks
            from .torch_dispatch import set_ep_group

            set_ep_group(ep_group)
            set_dispatch_chunks(chunks)
            self._register_checkpoint_saves(cfg)
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
        if self._uses_torch_backend(cfg):
            self._install_required_checkpoint_policy(cfg, model)

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

        # PEFT prefixes parameter names and ParamWrapper inserts `base_layer`
        # segments, so resolve by object identity; a name DDP cannot match is
        # broadcast from rank 0 and silently overwrites the rank's expert shard
        ignored_ids = {id(p) for p in getattr(inner, "_ep_ignored_params", [])}
        resolved = [
            n
            for n, p in list(model.named_parameters()) + list(model.named_buffers())
            if id(p) in ignored_ids
        ]
        if len(resolved) != len(ignored_ids):
            raise RuntimeError(
                f"expert_parallel: resolved {len(resolved)} of {len(ignored_ids)} "
                "EP-sharded expert parameters on the wrapped model; DDP would "
                "broadcast the unresolved ones from rank 0 and corrupt the expert "
                "shards."
            )

        # read the wrapper's own attribute: PeftModel.__getattr__ would forward
        # to the inner model and return the pre-wrap names again
        existing = list(model.__dict__.get("_ddp_params_and_buffers_to_ignore", []))
        model._ddp_params_and_buffers_to_ignore = existing + [
            n for n in resolved if n not in existing
        ]
        LOG.debug(
            f"expert_parallel: propagated {len(resolved)} DDP-ignored param "
            f"name(s) onto outer wrapper {type(model).__name__}."
        )

    @staticmethod
    def _register_padding_dispatch_hook(model) -> None:
        """Feed the batch's real-token mask to the EP dispatch so padding tokens are
        not routed (they'd otherwise pile onto one expert and break intranode dispatch).

        A model-level forward pre-hook reads the 2D ``attention_mask`` (1=real, 0=pad) and
        stashes a flattened ``[B*S]`` bool mask; ``_ep_forward`` sentinels those rows.
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
    def _register_checkpoint_saves(cfg) -> None:
        """Make every selective-checkpointing policy replay the forward's routing.

        The all-to-all split sizes come from ``topk``; a recompute that re-ran it could
        break near-ties differently on one rank and desync the collectives (a hang), and
        re-running the split's device->host copy would add a sync per layer.
        """
        from axolotl.monkeypatch.selective_checkpointing import (
            register_mandatory_save,
            register_preferred_save,
        )

        register_mandatory_save(ops={"aten::topk"}, cpu_copies=True)
        if getattr(cfg, "expert_parallel_save_dispatch", False):
            register_preferred_save(
                ops={
                    "axolotl::ep_all_to_all_single",
                    "axolotl::ep_all_to_all_single_equal",
                    "_c10d_functional::all_to_all_single",
                    "_c10d_functional::wait_tensor",
                }
            )

    @staticmethod
    def _install_required_checkpoint_policy(cfg, model) -> None:
        """Without ``selective_checkpointing``, still checkpoint through a policy that
        holds the registered routing saves (and nothing else). FSDP2
        ``activation_checkpointing`` picks the saves up in its own checkpoint wrapper."""
        if not getattr(cfg, "gradient_checkpointing", None) or getattr(
            cfg, "selective_checkpointing", None
        ):
            return
        from axolotl.monkeypatch.selective_checkpointing import (
            apply_selective_checkpointing,
        )

        # PeftModel forwards gradient_checkpointing_enable to the base model, so
        # wrapping the base covers callers holding either
        get_base_model = getattr(model, "get_base_model", None)
        base = get_base_model() if callable(get_base_model) else model
        apply_selective_checkpointing(base, save=[])

    @classmethod
    def _uses_torch_backend(cls, cfg) -> bool:
        backend = getattr(cfg, "expert_parallel_backend", None) or "auto"
        if backend == "auto":
            backend = cls._resolve_backend(cfg)
        return backend == "torch"

    @staticmethod
    def _infer_local_kernel(cfg) -> str:
        """Decide which local-experts kernel runs under EP dispatch.

        `use_scattermoe` / `use_sonicmoe` are the master flags from
        `kernels/args.py` and take precedence; otherwise fall back to
        `experts_implementation` (`eager` / `grouped_mm` / `batched_mm`, or an
        explicit `deep_ep_*` / `torch_ep_*` composite, which keeps its local kernel).
        """
        if getattr(cfg, "use_scattermoe", False):
            return "scattermoe"

        if getattr(cfg, "use_sonicmoe", False):
            return "sonicmoe"

        ei = getattr(cfg, "experts_implementation", None)
        if ei == "deep_ep":
            return "eager"
        for prefix in ("deep_ep_", "torch_ep_"):
            if isinstance(ei, str) and ei.startswith(prefix):
                return ei[len(prefix) :]
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
        if not dist.is_available():
            return None
        if not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            # pure EP builds no mesh, so nothing has created the process group yet
            from axolotl.utils.distributed import init_distributed_state

            init_distributed_state()
        if not dist.is_initialized():
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

    @classmethod
    def _resolve_backend(cls, cfg) -> str | None:
        """Resolve ``expert_parallel_backend`` to ``"deep_ep"`` / ``"torch"`` and store it on cfg.

        ``auto`` picks DeepEP when importable, else torch. Returns ``None`` when an explicit
        ``deep_ep`` is unavailable and the fallback is enabled (EP is then skipped)."""
        backend = getattr(cfg, "expert_parallel_backend", None) or "auto"
        if backend == "auto":
            backend = "deep_ep" if find_spec("deep_ep") is not None else "torch"
            cfg.expert_parallel_backend = backend
        if backend == "deep_ep" and not cls._deep_ep_available(cfg):
            return None
        return backend

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
