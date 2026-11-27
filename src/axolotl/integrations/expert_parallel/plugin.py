# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-Parallel plugin for axolotl (DeepEP, torch all-to-all or optmoe)."""

from __future__ import annotations

import math
import os
import time
from importlib.util import find_spec
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.utils.distributed import resolve_cfg_mesh
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# pre-``expert_parallel`` composite names -> local implementation; the backend prefix is ignored
# (as it always was), so ``expert_parallel_backend`` alone picks the dispatch
DEPRECATED_EXPERTS_IMPLEMENTATIONS = {
    "deep_ep": "eager",
    "deep_ep_grouped_mm": "grouped_mm",
    "deep_ep_scattermoe": "scattermoe",
    "deep_ep_sonicmoe": "sonicmoe",
    "torch_ep_eager": "eager",
    "torch_ep_grouped_mm": "grouped_mm",
    "torch_ep_scattermoe": "scattermoe",
    "torch_ep_sonicmoe": "sonicmoe",
}

KERNEL_IMPLEMENTATION_FLAGS = {
    "scattermoe": "use_scattermoe",
    "sonicmoe": "use_sonicmoe",
}

# transformers' deepgemm forward writes into torch.empty outputs with no autograd path
NO_BACKWARD_EXPERTS_IMPLEMENTATIONS = frozenset({"deepgemm"})

OPTMOE = "optmoe"
OPTMOE_LORA_FORWARD_OPS = ("optmoe::lora_forward", "optmoe::nvfp4_lora_forward")
_TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_topk", "top_k")


class OptMoEGeometry(NamedTuple):
    num_experts: int
    hidden: int
    intermediate: int
    top_k: int
    activation_clamp: float | None
    precision: str | None = None
    """The precision the model resolved (``nvfp4_lora``), ``None`` when the config's stands."""


class OptMoEDispatchGeometry(NamedTuple):
    hidden: int
    top_k: int


def expert_fsdp_mesh(mesh):
    """The mesh the routed experts FSDP-shard on under EP composition, or ``None``.

    Every non-``ep`` data axis (``dp_shard`` and ``cp``) is flattened into one shard
    dim: cp ranks hold different halves of the same sequences, so their expert grads must
    be reduced exactly like dp_shard's. ``dp_replicate`` stays a separate outer dim (HSDP).
    """
    names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
    if expert_shard_axis(names) is None:
        return None
    axes = [a for a in ("dp_shard", "cp") if a in names]
    if len(axes) == 1:
        shard_name = axes[0]
    else:
        shard_name = "ep_shard"
        if shard_name not in (mesh._get_root_mesh().mesh_dim_names or ()):
            mesh[tuple(axes)]._flatten(shard_name)
    if "dp_replicate" in names:
        return mesh[("dp_replicate", shard_name)]
    return mesh[shard_name]


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


def per_rank_expert_mesh(mesh, device_type: str):
    """The mesh the routed experts FSDP-wrap on when EP has no shard axis (pure EP, or
    ``dp_replicate x ep``): a fresh ``(dp_replicate?, ep, dp_shard=1)`` mesh whose size-1
    ``dp_shard`` dim keeps each rank's expert slice whole (the all-gather is a no-op) while
    making every param a DTensor; ``dp_replicate`` stays the outer dim so HSDP reduces the
    expert grads across replicas. Returns the ``(dp_replicate, dp_shard)`` or ``dp_shard``
    sub-mesh to hand ``fully_shard_experts``."""
    from torch.distributed.device_mesh import init_device_mesh

    names = (
        tuple(getattr(mesh, "mesh_dim_names", None) or ()) if mesh is not None else ()
    )
    if "ep" in names:
        ep_size = mesh["ep"].size()
        replicate = mesh["dp_replicate"].size() if "dp_replicate" in names else 1
    else:
        ep_size, replicate = dist.get_world_size(), 1
    shape: tuple[int, ...] = (ep_size, 1)
    dim_names: tuple[str, ...] = ("ep", "dp_shard")
    if replicate > 1:
        shape, dim_names = (replicate, *shape), ("dp_replicate", *dim_names)
    root = init_device_mesh(device_type, shape, mesh_dim_names=dim_names)
    return root[("dp_replicate", "dp_shard")] if replicate > 1 else root["dp_shard"]


def _apply_gate_is_plain(experts) -> bool:
    """Whether an overridden ``_apply_gate`` still computes ``act_fn(gate) * up``.

    Beam routes the plain SwiGLU through an FP32 override; a numeric probe tells
    that apart from a different gating function without naming model types.
    """
    act_fn = getattr(experts, "act_fn", None)
    if act_fn is None:
        return False
    # Span far past any plausible clamp so a clamped or saturating gate cannot pass by luck.
    tail = torch.logspace(1, 4, 16, dtype=torch.float32)
    probe = torch.cat(
        [torch.linspace(-8.0, 8.0, 32, dtype=torch.float32), -tail, tail]
    ).reshape(2, 32)
    try:
        with torch.no_grad():
            got = experts._apply_gate(probe)
    except (RuntimeError, TypeError, ValueError, AttributeError, IndexError):
        return False
    gate, up = probe.chunk(2, dim=-1)
    want = act_fn(gate) * up
    return tuple(got.shape) == tuple(want.shape) and torch.allclose(
        got.float(), want, atol=1e-5, rtol=1e-4
    )


class _OptMoEPrepareCallback(TrainerCallback):
    """Compiles the fused ``optmoe`` kernels on every rank before the first step, after the
    model is wrapped (``OptMoERuntime.prepare``)."""

    def __init__(self, plugin, cfg, trainer):
        self.plugin, self.cfg, self.trainer = plugin, cfg, trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.plugin._prepare_optmoe_runtime(self.cfg, self.trainer)


class ExpertParallelPlugin(BasePlugin):
    """Plugin that swaps MoE dispatch/combine for expert-parallel token dispatch."""

    def get_input_args(self):
        return "axolotl.integrations.expert_parallel.ExpertParallelArgs"

    _optmoe_geometry: OptMoEGeometry | None = None
    _optmoe_dispatch_geometry: OptMoEDispatchGeometry | None = None

    def get_distributed_env(self, cfg) -> dict[str, str]:
        """``NCCL_MAX_CTAS`` sized to the SMs ``optmoe`` leaves free, so FSDP collectives
        cannot deadlock behind the persistent kernels. Pure: no CUDA or process group.
        Nothing is exported once the config alone rules the fused mode out."""
        if not self._is_ep_enabled(cfg) or not self._optmoe_fused_possible(cfg):
            return {}
        from optmoe import DEFAULT_RESERVED_SMS, nccl_env

        return nccl_env(
            reserved_sms=DEFAULT_RESERVED_SMS,
            dp_replicate=getattr(cfg, "dp_replicate_size", None) or 1,
        )

    def pre_model_load(self, cfg):
        if not self._is_ep_enabled(cfg):
            return

        self._migrate_deprecated_experts_implementation(cfg)
        backend = self._resolve_backend(cfg)
        if backend is None:
            return  # already-warned fallback path

        # Cross-cfg validation that args.py can't do (it only sees its own fields).
        self._validate_mesh_axes(cfg)
        self._reject_transformers_distribution(cfg)

        from .experts_fn import (
            EXPERT_PARALLEL,
            register_all,
            set_backend,
            set_local_implementation,
        )

        register_all()

        # optmoe's fused mode ignores the local implementation, but its dispatch mode
        # and a later model-level fallback to torch run it, so it is validated and
        # registered all the same
        local = self._infer_local_implementation(cfg)
        self._check_local_implementation(local)
        self._register_kernel_implementation(local)
        set_backend(backend)
        set_local_implementation(local)
        mode = None
        if backend == OPTMOE:
            from .experts_fn import set_optmoe_mode

            mode = self._optmoe_mode(cfg)
            fused_possible = self._optmoe_fused_possible(cfg)
            set_optmoe_mode("fused" if fused_possible else "dispatch")
            if fused_possible:
                from optmoe import DEFAULT_RESERVED_SMS, check_nccl_env

                check_nccl_env(
                    reserved_sms=DEFAULT_RESERVED_SMS,
                    dp_replicate=getattr(cfg, "dp_replicate_size", None) or 1,
                )
        previous = getattr(cfg, "experts_implementation", None)
        cfg.experts_implementation = EXPERT_PARALLEL
        if backend != OPTMOE:
            detail = f"wrapping {local!r}"
        elif mode == "fused":
            detail = f"(fused; {local!r} only on fallback)"
        elif mode == "dispatch":
            detail = f"(peer dispatch wrapping {local!r})"
        else:
            detail = f"(mode auto: fused, else peer dispatch wrapping {local!r})"
        LOG.info(
            f"expert_parallel: backend={backend!r}, experts_implementation "
            f"{previous!r} -> {EXPERT_PARALLEL!r} {detail}"
        )

    def post_model_build(self, cfg, model):
        if not self._is_ep_enabled(cfg):
            return
        backend = self._resolve_backend(cfg)
        if backend is None:
            return

        from .shard import shard_expert_weights

        self._reject_transformers_distributed_model(model)
        if backend == OPTMOE:
            backend = self._resolve_optmoe_model(cfg, model)
        ep_group = self._resolve_ep_group(cfg)
        ep_world = dist.get_world_size(ep_group) if ep_group is not None else 1
        # rank zero streams every expert slice to its peers here; on a 500B MoE this
        # takes one to five minutes with no other output
        LOG.info(
            f"Sharding expert weights across {ep_world} expert-parallel rank(s)..."
        )
        sharded = shard_expert_weights(model, ep_group)

        if sharded == 0:
            if ep_world <= 1:
                world = dist.get_world_size() if dist.is_initialized() else 1
                LOG.warning(
                    f"expert_parallel_size={cfg.expert_parallel_size} but this process resolved "
                    f"an EP group of size {ep_world} (world_size={world}); expert weights stay "
                    "unsharded and dispatch/combine is a no-op. Launch with `accelerate launch` "
                    "or `torchrun` so every EP rank is a separate process."
                )
            else:
                raise ValueError(
                    "expert_parallel_size > 1 but no Experts modules were detected for "
                    "sharding (the model does not use transformers' canonical 3-D "
                    "gate_up_proj (or non-gated up_proj)/down_proj experts layout). "
                    "Expert parallelism cannot run "
                    "on this model; set expert_parallel_size: 1."
                )
        else:
            LOG.info(
                f"expert_parallel: sharded {sharded} Experts module(s); "
                f"ep_rank={dist.get_rank(ep_group)}/{ep_world}"
            )

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
        elif backend == OPTMOE:
            self._register_checkpoint_saves(cfg, model)
        else:
            from .experts_fn import set_dispatch_chunks
            from .torch_dispatch import set_ep_group

            set_ep_group(ep_group)
            set_dispatch_chunks(chunks)
            self._register_checkpoint_saves(cfg, model)
        from .experts_fn import set_backend, set_token_capacity

        set_backend(backend)
        set_token_capacity(getattr(cfg, "expert_parallel_token_capacity", None))
        # Pure-EP path: register the grad-scale hook now. FSDP+EP defers
        # registration to `fully_shard_experts` (after experts become DTensors).
        mesh = resolve_cfg_mesh(cfg)
        if mesh.efsdp <= 1:
            self._register_expert_grad_scale(model, cfg.expert_parallel_size or 1)

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
        if getattr(cfg, "expert_parallel_backend", None) == OPTMOE:
            if self._optmoe_mode(cfg) == "dispatch":
                self._build_optmoe_dispatch_runtime(cfg)
            else:
                self._build_optmoe_runtime(cfg, model)
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
        get_base_model = getattr(model, "get_base_model", None)
        base = get_base_model() if callable(get_base_model) else model
        if not getattr(base, "moe_tokens_match_input_ids", True):
            # the model reshapes tokens before its MoE layers (e.g. TwoTower's
            # per-document prompts and canvases): an input_ids mask cannot line up
            set_valid_token_mask(None)
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
    def _routing_blocks(model) -> list[torch.nn.Module]:
        """The module directly holding each Experts module: the MoE block whose forward
        runs the router. Empty when any Experts module has no such block (or it is the
        model itself, which runs outside the checkpointed layers)."""
        from .shard import _detect_experts_modules, _is_param_wrapper

        by_name = dict(model.named_modules())
        blocks: dict[int, torch.nn.Module] = {}
        for name, _module in _detect_experts_modules(model):
            parent_name = name.rpartition(".")[0]
            while parent_name and _is_param_wrapper(by_name[parent_name]):
                parent_name = parent_name.rpartition(".")[0]
            if not parent_name:
                return []
            blocks.setdefault(id(by_name[parent_name]), by_name[parent_name])
        return list(blocks.values())

    @classmethod
    def _register_checkpoint_saves(cls, cfg, model=None) -> None:
        """Make every selective-checkpointing policy replay the forward's routing.

        The all-to-all split sizes come from ``topk``; a recompute that re-ran it could
        break near-ties differently on one rank and desync the collectives (a hang), and
        re-running the split's device->host copy would add a sync per layer. The
        ``topk`` save is scoped to the MoE blocks so a ``topk`` elsewhere in the layer
        (an attention indexer's) is recomputed rather than held to backward.
        """
        from axolotl.monkeypatch.selective_checkpointing import (
            register_mandatory_save,
            register_preferred_save,
        )

        # the optmoe compact dispatch has the same one host count per layer
        register_mandatory_save(
            ops={"axolotl::ep_to_host", "optmoe::peer_count_to_host"}
        )
        routing_blocks = cls._routing_blocks(model) if model is not None else []
        if routing_blocks:
            register_mandatory_save(ops={"aten::topk"}, within=routing_blocks)
        else:
            register_mandatory_save(ops={"aten::topk"})
        if getattr(cfg, "expert_parallel_save_dispatch", False):
            # wait_tensor aliases the collective's storage; a replayed wait is a no-op
            register_preferred_save(
                ops={
                    "axolotl::ep_all_to_all_single",
                    "axolotl::ep_all_to_all_single_equal",
                    "_c10d_functional::all_to_all_single",
                }
            )

    @staticmethod
    def _install_required_checkpoint_policy(cfg, model) -> None:
        """Without ``activation_checkpointing.selective``, still checkpoint through a
        policy that holds the registered routing saves (and nothing else). FSDP2's
        checkpoint wrapper (``implementation: fsdp``) picks the saves up on its own."""
        from axolotl.utils.activation_checkpointing import (
            checkpointing_config,
            hf_checkpointing_enabled,
        )

        if (
            not hf_checkpointing_enabled(cfg)
            or checkpointing_config(cfg).selective is not None
        ):
            return
        from axolotl.monkeypatch.selective_checkpointing import (
            apply_selective_checkpointing,
        )

        # PeftModel forwards the checkpointing enable call to the base model, so
        # wrapping the base covers callers holding either
        get_base_model = getattr(model, "get_base_model", None)
        base = get_base_model() if callable(get_base_model) else model
        apply_selective_checkpointing(base, save=[])

    @classmethod
    def _uses_torch_backend(cls, cfg) -> bool:
        """Whether the backend needs the routing replayed under checkpointing (torch's
        collectives; optmoe's saved fused output)."""
        backend = getattr(cfg, "expert_parallel_backend", None) or "auto"
        if backend == "auto":
            backend = cls._resolve_backend(cfg)
        return backend in ("torch", OPTMOE)

    def add_callbacks_post_trainer(self, cfg, trainer):
        runtime = self.__dict__.get("_optmoe_runtime")
        if runtime is None or not hasattr(runtime, "prepare"):
            return []
        return [_OptMoEPrepareCallback(self, cfg, trainer)]

    def _prepare_optmoe_runtime(self, cfg, trainer) -> None:
        from .shard import _detect_experts_modules

        runtime = self._optmoe_runtime
        model = trainer.model
        experts = next((m for _n, m in _detect_experts_modules(model)), None)
        weight = getattr(experts, "gate_up_proj", None)
        if not isinstance(weight, torch.Tensor):
            weight = None
        lora_factor = next(
            (
                p
                for n, p in model.named_parameters()
                if "lora_A" in n and "experts" in n
            ),
            None,
        )
        # the dtype the fused forward sees: FSDP2 casts to the mixed-precision param dtype
        # unless the LoRA FP32-gradient policy keeps the loaded dtypes
        fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
        policy = getattr(fsdp_plugin, "mixed_precision_policy", None)
        cast = (
            getattr(fsdp_plugin, "fsdp_version", 1) == 2
            and getattr(policy, "param_dtype", None) is not None
            and not getattr(
                trainer.accelerator.unwrap_model(model),
                "_axolotl_lora_fp32_gradients",
                False,
            )
        )

        def forward_dtype(param, default):
            if cast:
                return policy.param_dtype
            return param.dtype if param is not None else default

        kwargs = {}
        lora_state: Any = None
        if runtime.precision.endswith("_lora"):
            kwargs = {
                "lora_rank": cfg.lora_r,
                "lora_dtype": forward_dtype(lora_factor, torch.float32),
            }
            if runtime.precision == "nvfp4_lora":
                from .nvfp4_lora import get_state

                lora_state = get_state()
                kwargs["nvfp4_backend"] = getattr(lora_state, "backend", None)
            else:
                from . import fused_lora

                lora_state = fused_lora.get_state()
        else:
            kwargs["weight_dtype"] = forward_dtype(weight, torch.bfloat16)
        tokens = min(self._optmoe_tokens_per_rank(cfg), runtime.max_tokens_per_rank)
        start = time.perf_counter()
        runtime.prepare(num_tokens=tokens, **kwargs)
        LOG.info(
            f"expert_parallel: optmoe kernels prepared on every rank in "
            f"{time.perf_counter() - start:.1f}s ({tokens} tokens/rank, {kwargs})"
        )
        if lora_state is not None:
            layers = lora_state.prepare_layers(model)
            LOG.info(f"expert_parallel: {layers} optmoe LoRA layers registered")

    def post_train_unload(self, cfg):
        from . import fused_lora
        from .experts_fn import set_optmoe_runtime
        from .nvfp4_lora import set_state

        set_state(None)
        fused_lora.set_state(None)
        runtime = self.__dict__.pop("_optmoe_runtime", None)
        set_optmoe_runtime(None)
        if runtime is not None:
            runtime.destroy()

    @staticmethod
    def _optmoe_mode(cfg) -> str:
        return getattr(cfg, "expert_parallel_optmoe_mode", None) or "auto"

    @classmethod
    def _optmoe_config_reason(cls, cfg) -> str | None:
        """Config-only reasons ``optmoe`` cannot run at all (pure: no CUDA). In the forced
        ``fused`` mode the fused mode's own config reasons count too."""
        if find_spec("optmoe") is None:
            return "the optmoe package is not installed (pip install optmoe)"
        if cls._optmoe_mode(cfg) == "fused":
            return cls._optmoe_fused_config_reason(cfg)
        return None

    @classmethod
    def _optmoe_fused_possible(cls, cfg) -> bool:
        """Whether the config still allows the fused mode (so the SM reservation applies)."""
        return (
            getattr(cfg, "expert_parallel_backend", None) == OPTMOE
            and cls._optmoe_mode(cfg) != "dispatch"
            and cls._optmoe_config_reason(cfg) is None
            and cls._optmoe_fused_config_reason(cfg) is None
        )

    @staticmethod
    def _optmoe_fused_config_reason(cfg) -> str | None:
        """Config-only reasons the fused mode cannot run (the dispatch mode can)."""
        from . import fused_lora
        from .nvfp4_lora import (
            PRECISION as NVFP4_LORA,
        )

        if find_spec("optmoe_deepgemm") is None:
            return (
                "the optmoe kernel extension is not installed "
                "(python -m optmoe.install_kernels)"
            )
        precision = getattr(cfg, "expert_parallel_optmoe_precision", None) or "auto"
        if getattr(cfg, "adapter", None):
            reason = ExpertParallelPlugin._optmoe_lora_config_reason(cfg, precision)
            if reason is not None:
                if precision == NVFP4_LORA or precision in fused_lora.PRECISIONS:
                    return reason
                return (
                    f"adapter {cfg.adapter!r} is configured; the fused kernels cannot "
                    f"train it ({reason})"
                )
        elif precision == NVFP4_LORA or precision in fused_lora.PRECISIONS:
            return f"precision {precision} needs adapter: lora on the experts"
        quantized = [
            key
            for key in (
                "load_in_4bit",
                "load_in_8bit",
                "gptq",
                "model_quantization_config",
            )
            if getattr(cfg, key, None)
        ]
        if quantized:
            return f"quantized model loading ({', '.join(quantized)}) is unsupported"
        return None

    @staticmethod
    def _optmoe_lora_route(cfg, precision: str) -> str | None:
        """Which fused path trains the adapter: ``nvfp4_lora`` (frozen packed NVFP4 experts;
        ``auto`` takes it when the config qualifies), ``fused_lora`` (merged BF16 experts on
        the configured precision's kernels), or ``None`` when neither applies."""
        from . import fused_lora
        from .nvfp4_lora import (
            PRECISION as NVFP4_LORA,
            base_is_nvfp4_modelopt,
            config_reason as nvfp4_lora_config_reason,
        )

        if precision == NVFP4_LORA:
            return NVFP4_LORA
        if precision == "auto" and nvfp4_lora_config_reason(cfg) is None:
            return NVFP4_LORA
        if not fused_lora.enabled(cfg) or base_is_nvfp4_modelopt(cfg):
            return None
        return "fused_lora"

    @staticmethod
    def _optmoe_lora_config_reason(cfg, precision: str) -> str | None:
        """Why the fused mode cannot train the configured adapter (``None`` = it can)."""
        from . import fused_lora
        from .nvfp4_lora import (
            PRECISION as NVFP4_LORA,
            base_is_nvfp4_modelopt,
            config_reason as nvfp4_lora_config_reason,
        )

        route = ExpertParallelPlugin._optmoe_lora_route(cfg, precision)
        if route == NVFP4_LORA:
            return nvfp4_lora_config_reason(cfg)
        if route == "fused_lora":
            return fused_lora.config_reason(cfg)
        if not fused_lora.enabled(cfg):
            return (
                "expert_parallel_optmoe_fused_lora is false, so precision "
                f"{precision} trains full expert weights only"
            )
        if base_is_nvfp4_modelopt(cfg) and precision == "auto":
            return nvfp4_lora_config_reason(cfg)
        return f"precision {precision} on a packed NVFP4 base trains LoRA only as nvfp4_lora"

    @staticmethod
    def _optmoe_fallback(cfg, reason: str) -> str:
        """Degrade to torch dispatch with the configured local kernel, or raise."""
        msg = f"expert_parallel_backend=optmoe is unsupported here: {reason}."
        if not getattr(cfg, "expert_parallel_fallback_on_unsupported", True):
            error = ImportError if "not installed" in reason else ValueError
            raise error(
                msg + " Set expert_parallel_fallback_on_unsupported: true to fall back "
                "to expert_parallel_backend=torch."
            )
        LOG.warning(
            msg + " Falling back to expert_parallel_backend=torch with the configured "
            "local experts implementation."
        )
        cfg.expert_parallel_backend = "torch"
        return "torch"

    @staticmethod
    def _expert_activation(experts) -> tuple[str, float | None, str | None]:
        """``(activation, clamp, reason)`` for a gated experts module's epilogue."""
        from transformers.activations import SiLUActivation

        for attr in ("alpha", "swiglu_alpha"):
            if getattr(experts, attr, None) is not None:
                return "custom", None, f"gated activation with {attr} is unsupported"
        clamp = getattr(experts, "limit", None)
        if clamp is None:
            clamp = getattr(experts, "swiglu_limit", None)
        apply_gate = getattr(type(experts), "_apply_gate", None)
        if apply_gate is not None and clamp is None:
            from transformers.integrations.moe import _default_apply_gate

            if apply_gate is not _default_apply_gate and not _apply_gate_is_plain(
                experts
            ):
                return "custom", None, "custom gated activation (_apply_gate)"
        act_fn = getattr(experts, "act_fn", None)
        silu = isinstance(act_fn, (torch.nn.SiLU, SiLUActivation))
        return ("silu" if silu else type(act_fn).__name__), clamp, None

    @classmethod
    def _optmoe_model_reason(
        cls, cfg, model
    ) -> tuple[OptMoEGeometry | None, str | None]:
        """The fused geometry shared by every Experts module, or why there is none."""
        from .experts_fn import _maybe_install_decorator_attrs
        from .shard import _detect_experts_modules

        if not torch.cuda.is_available():
            return None, "CUDA is unavailable"
        capability = torch.cuda.get_device_capability()
        if capability < (9, 0):
            return None, f"compute capability sm_{capability[0]}{capability[1]} < SM90"
        config = getattr(model, "config", None)
        get_text_config = getattr(config, "get_text_config", None)
        text_config = get_text_config() if callable(get_text_config) else config
        top_k = next(
            (
                getattr(text_config, key)
                for key in _TOP_K_KEYS
                if isinstance(getattr(text_config, key, None), int)
            ),
            None,
        )
        if top_k is None:
            return None, "the model config does not name its router top-k"
        geometries = set()
        for name, experts in _detect_experts_modules(model):
            _maybe_install_decorator_attrs(experts)
            gate_up = getattr(experts, "gate_up_proj", None)
            down = getattr(experts, "down_proj", None)
            if (
                gate_up is None
                or down is None
                or gate_up.dim() != 3
                or down.dim() != 3
                or not experts.has_gate
            ):
                return None, f"{name}: experts are not gated gate_up_proj/down_proj"
            if experts.is_transposed or not getattr(experts, "is_concatenated", True):
                return None, f"{name}: transposed or interleaved expert weights"
            if experts.has_post_expert_norm:
                return None, f"{name}: post-expert normalization is unsupported"
            num_experts, two_i, hidden = gate_up.shape
            intermediate = down.shape[-1]
            if two_i != 2 * intermediate or tuple(down.shape) != (
                num_experts,
                hidden,
                intermediate,
            ):
                return None, f"{name}: gate_up_proj/down_proj shapes do not match"
            activation, clamp, reason = cls._expert_activation(experts)
            if reason is not None:
                return None, f"{name}: {reason}"
            precision = getattr(cfg, "expert_parallel_optmoe_precision", None) or "auto"
            resolved = None
            if getattr(cfg, "adapter", None):
                from . import fused_lora
                from .nvfp4_lora import PRECISION as NVFP4_LORA, experts_reason

                # the config already qualified for a fused LoRA route; the experts must be
                # what that route's kernels consume
                if cls._optmoe_lora_route(cfg, precision) == NVFP4_LORA:
                    reason = experts_reason(experts, name)
                    resolved = NVFP4_LORA
                else:
                    from optmoe import lora_precision

                    reason = fused_lora.experts_reason(experts, name)
                    try:
                        resolved = lora_precision(precision, capability)
                    except ValueError as exc:
                        return None, str(exc)
                    if reason is None and (hidden % 256 or intermediate % 256):
                        # the fused LoRA precisions run only on the persistent backward
                        reason = (
                            f"{name}: fused LoRA needs hidden and intermediate multiples "
                            f"of 256, got {hidden} and {intermediate}"
                        )
                if reason is not None:
                    return None, reason
                precision = resolved
            from optmoe import unsupported_reason

            reason = unsupported_reason(
                capability=capability,
                hidden=hidden,
                intermediate=intermediate,
                has_bias=bool(experts.has_bias),
                activation=activation,
                activation_clamp=clamp,
                precision=precision,
            )
            if reason is not None:
                return None, f"{name}: {reason}"
            geometries.add(
                OptMoEGeometry(
                    num_experts, hidden, intermediate, top_k, clamp, resolved
                )
            )
        if len(geometries) != 1:
            return None, (
                "no Experts modules found"
                if not geometries
                else "Experts modules differ in geometry; one runtime serves every layer"
            )
        return geometries.pop(), None

    @classmethod
    def _optmoe_dispatch_reason(
        cls, cfg, model
    ) -> tuple[OptMoEDispatchGeometry | None, str | None]:
        """The dispatch mode's geometry, or why the peer dispatch cannot run here. The
        single-host requirement is checked by the runtime on the resolved EP group."""
        if not torch.cuda.is_available():
            return None, "CUDA is unavailable"
        from optmoe import dispatch_unsupported_reason

        reason = dispatch_unsupported_reason(
            capability=torch.cuda.get_device_capability()
        )
        if reason is not None:
            return None, reason
        config = getattr(model, "config", None)
        get_text_config = getattr(config, "get_text_config", None)
        text_config = get_text_config() if callable(get_text_config) else config
        hidden = getattr(text_config, "hidden_size", None)
        top_k = next(
            (
                getattr(text_config, key)
                for key in _TOP_K_KEYS
                if isinstance(getattr(text_config, key, None), int)
            ),
            None,
        )
        if not isinstance(hidden, int) or top_k is None:
            return (
                None,
                "the model config does not name its hidden size and router top-k",
            )
        return OptMoEDispatchGeometry(hidden, top_k), None

    def _resolve_optmoe_model(self, cfg, model) -> str:
        """Settle ``expert_parallel_optmoe_mode`` against the model: ``fused`` when it is
        allowed and supported, else ``dispatch``, else the backend fallback."""
        from .experts_fn import set_backend, set_optmoe_mode

        mode = self._optmoe_mode(cfg)
        fused_reason = None
        if mode != "dispatch":
            fused_reason = self._optmoe_fused_config_reason(cfg)
            if fused_reason is None:
                geometry, fused_reason = self._optmoe_model_reason(cfg, model)
            if fused_reason is None:
                self._optmoe_geometry = geometry
                cfg.expert_parallel_optmoe_mode = "fused"
                if geometry is not None and geometry.precision is not None:
                    cfg.expert_parallel_optmoe_precision = geometry.precision
                set_optmoe_mode("fused")
                return OPTMOE
            if mode == "fused":
                backend = self._optmoe_fallback(cfg, fused_reason)
                set_backend(backend)
                return backend
            LOG.info(
                f"expert_parallel: optmoe fused mode is unsupported here ({fused_reason}); "
                "using the peer dispatch mode with the configured local experts kernel."
            )
        dispatch_geometry, reason = self._optmoe_dispatch_reason(cfg, model)
        if reason is not None:
            backend = self._optmoe_fallback(cfg, f"peer dispatch: {reason}")
            set_backend(backend)
            return backend
        self._optmoe_dispatch_geometry = dispatch_geometry
        cfg.expert_parallel_optmoe_mode = "dispatch"
        set_optmoe_mode("dispatch")
        return OPTMOE

    def _build_optmoe_dispatch_runtime(self, cfg) -> None:
        """One peer dispatch/combine runtime per process, shared by every MoE layer."""
        from optmoe import OptMoEDispatchRuntime

        from axolotl.monkeypatch.selective_checkpointing import register_preferred_save

        from .experts_fn import set_optmoe_runtime

        geometry = self._optmoe_dispatch_geometry
        if geometry is None:
            raise RuntimeError(
                "expert_parallel: optmoe dispatch geometry was not resolved in "
                "post_model_build"
            )
        if self.__dict__.get("_optmoe_runtime") is not None:
            return
        compact = bool(getattr(cfg, "expert_parallel_optmoe_compact", False))
        bounded = compact and bool(
            getattr(cfg, "expert_parallel_optmoe_bounded", False)
        )
        runtime = OptMoEDispatchRuntime(
            group=self._resolve_ep_group(cfg),
            max_tokens_per_rank=self._optmoe_tokens_per_rank(cfg),
            top_k=geometry.top_k,
            hidden=geometry.hidden,
            compact=compact,
            bounded=bounded,
        )
        self._optmoe_runtime = runtime
        set_optmoe_runtime(runtime)
        if getattr(cfg, "expert_parallel_save_dispatch", False):
            # same trade as the torch all-to-all saves: hold the received rows so the
            # recompute does not re-run the peer transfers
            register_preferred_save(
                ops=getattr(
                    runtime, "save_ops", {runtime.dispatch_op, runtime.combine_op}
                )
            )
        LOG.info(
            f"expert_parallel: optmoe peer dispatch runtime, "
            f"{runtime.max_tokens_per_rank} tokens/rank, {runtime.receive_rows} receive "
            f"rows ({'routed rows only' if compact else 'fixed layout'}"
            f"{', bounded receive' if bounded else ''}), "
            f"top-{geometry.top_k}, hidden {geometry.hidden}"
        )

    @staticmethod
    def _optmoe_tokens_per_rank(cfg) -> int:
        batch = max(
            getattr(cfg, "micro_batch_size", None) or 1,
            getattr(cfg, "eval_batch_size", None) or 1,
        )
        return math.ceil(
            cfg.sequence_len
            * batch
            / (getattr(cfg, "context_parallel_size", None) or 1)
        )

    def _build_optmoe_runtime(self, cfg, model=None) -> None:
        """One runtime (and symmetric buffer) per process, shared by every MoE layer."""
        from optmoe import DEFAULT_RESERVED_SMS, OptMoERuntime

        from axolotl.monkeypatch.selective_checkpointing import register_mandatory_save

        from .experts_fn import set_optmoe_runtime

        geometry = self._optmoe_geometry
        if geometry is None:
            raise RuntimeError(
                "expert_parallel: optmoe geometry was not resolved in post_model_build"
            )
        if self.__dict__.get("_optmoe_runtime") is not None:
            return
        os.environ["OPTMOE_WORKSPACE_BOUND"] = (
            getattr(cfg, "expert_parallel_optmoe_workspace_bound", None) or "auto"
        )
        os.environ["OPTMOE_STORED_CONTEXT"] = (
            getattr(cfg, "expert_parallel_optmoe_stored_context", None) or "auto"
        )
        os.environ["OPTMOE_NVFP4_MERGED_CACHE"] = (
            getattr(cfg, "expert_parallel_optmoe_nvfp4_merged_cache", None) or "layer"
        )
        os.environ["OPTMOE_NVFP4_LORA_HEADROOM_LOG2"] = str(
            getattr(cfg, "expert_parallel_optmoe_nvfp4_headroom_log2", None) or 0
        )
        tokens = self._optmoe_tokens_per_rank(cfg)
        alignment = OptMoERuntime.token_alignment()
        tokens = -(-tokens // alignment) * alignment
        runtime = OptMoERuntime(
            group=self._resolve_ep_group(cfg),
            num_experts=geometry.num_experts,
            max_tokens_per_rank=tokens,
            top_k=geometry.top_k,
            hidden=geometry.hidden,
            intermediate=geometry.intermediate,
            precision=getattr(cfg, "expert_parallel_optmoe_precision", None) or "auto",
            reserved_sms=DEFAULT_RESERVED_SMS,
            activation_clamp=geometry.activation_clamp,
        )
        self._optmoe_runtime = runtime
        set_optmoe_runtime(runtime)
        # the op saves only its inputs and recomputes in its own backward; SAC must keep
        # its output instead of replaying a collective kernel
        register_mandatory_save(ops={runtime.forward_op, *OPTMOE_LORA_FORWARD_OPS})
        precision = getattr(runtime, "precision", None) or getattr(
            cfg, "expert_parallel_optmoe_precision", None
        )
        from . import fused_lora
        from .nvfp4_lora import PRECISION as NVFP4_LORA

        if precision == NVFP4_LORA:
            from .nvfp4_lora import Nvfp4LoraState, set_state, tag_layers

            if model is not None:
                tag_layers(model)
            set_state(
                Nvfp4LoraState(
                    repo_id=getattr(cfg, "base_model", None),
                    revision=getattr(cfg, "revision_of_model", None),
                )
            )
        elif precision in fused_lora.PRECISIONS:
            os.environ["OPTMOE_LORA_MERGED_CACHE"] = (
                getattr(cfg, "expert_parallel_optmoe_lora_merged_cache", None)
                or "layer"
            )
            fused_lora.set_state(fused_lora.FusedLoraState())
        LOG.info(
            f"expert_parallel: optmoe runtime, {tokens} tokens/rank, "
            f"{geometry.num_experts} experts, top-{geometry.top_k}, "
            f"precision={getattr(runtime, 'precision', '?')}"
        )

    @staticmethod
    def _migrate_deprecated_experts_implementation(cfg) -> None:
        """Rewrite a pre-``expert_parallel`` composite name to its local implementation.

        A scattermoe/sonicmoe composite also sets its ``use_*`` flag (unless one is already
        set), which is what selects that kernel now."""
        name = getattr(cfg, "experts_implementation", None)
        if name not in DEPRECATED_EXPERTS_IMPLEMENTATIONS:
            return
        local = DEPRECATED_EXPERTS_IMPLEMENTATIONS[name]
        flag = KERNEL_IMPLEMENTATION_FLAGS.get(local)
        replacement = f"{flag}: true" if flag else f"experts_implementation: {local}"
        LOG.warning(
            f"experts_implementation: {name} is deprecated; use {replacement} "
            "(the expert_parallel plugin wraps it on whichever expert_parallel_backend "
            "is configured; the name's backend prefix is ignored)."
        )
        if flag and not any(
            getattr(cfg, f, False) for f in KERNEL_IMPLEMENTATION_FLAGS.values()
        ):
            setattr(cfg, flag, True)
        cfg.experts_implementation = local

    @staticmethod
    def _infer_local_implementation(cfg) -> str:
        """Decide which experts implementation runs on the dispatched rows.

        `use_scattermoe` / `use_sonicmoe` are the master flags from
        `kernels/args.py` and take precedence; otherwise `experts_implementation`
        names any registered implementation, defaulting to `grouped_mm`.
        """
        if getattr(cfg, "use_scattermoe", False):
            return "scattermoe"

        if getattr(cfg, "use_sonicmoe", False):
            return "sonicmoe"

        from .experts_fn import EXPERT_PARALLEL

        ei = getattr(cfg, "experts_implementation", None)
        if ei in ("scattermoe", "sonicmoe"):
            LOG.warning(
                f"expert_parallel: experts_implementation: {ei} needs use_{ei}: true "
                "(the kernels plugin registers and configures it); using grouped_mm."
            )
            return "grouped_mm"
        if ei in (None, EXPERT_PARALLEL):
            return "grouped_mm"
        return str(ei)

    @staticmethod
    def _check_local_implementation(name: str) -> None:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        if name in NO_BACKWARD_EXPERTS_IMPLEMENTATIONS:
            raise ValueError(
                f"expert_parallel: experts_implementation {name!r} has no backward pass "
                "(expert weights would silently not train); pick grouped_mm, batched_mm, "
                "eager, use_scattermoe or use_sonicmoe."
            )
        if name in ("eager", "scattermoe", "sonicmoe") or name in ALL_EXPERTS_FUNCTIONS:
            return
        raise ValueError(
            f"expert_parallel: experts_implementation {name!r} is not registered in "
            "transformers' ALL_EXPERTS_FUNCTIONS; register it before model load or "
            "pick one of eager, "
            f"{', '.join(sorted(ALL_EXPERTS_FUNCTIONS.keys()))}."
        )

    @staticmethod
    def _register_kernel_implementation(name: str) -> None:
        """Register axolotl's scattermoe/sonicmoe forward, whichever plugin runs first;
        otherwise the name would resolve to nothing (scattermoe) or to transformers'
        LoRA-unaware sonicmoe."""
        if name == "scattermoe":
            from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
                register_scattermoe_experts,
            )

            register_scattermoe_experts()
        elif name == "sonicmoe":
            from axolotl.integrations.kernels.libs.sonicmoe.experts import (
                register_sonicmoe_experts,
            )

            register_sonicmoe_experts()

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

        # A bare AcceleratorState() before any Accelerator() writes part of the shared state and
        # raises; a later Accelerator() then takes it as initialized and never sets mixed_precision.
        if "_mixed_precision" not in AcceleratorState._shared_state:
            LOG.debug("expert_parallel: no Accelerator yet, so no accelerate mesh")
            return None
        mesh = AcceleratorState._shared_state.get("device_mesh")
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
        if (getattr(cfg, "expert_parallel_size", 1) or 1) <= 1:
            return dist.group.WORLD
        ep_size, dp_replicate_size, dp_shard_size, tp_size, cp_size = (
            ExpertParallelPlugin._expert_layout(cfg, world_size)
        )

        if ep_size == world_size:
            LOG.info(f"expert_parallel: pure EP, ep group = WORLD (ep={ep_size})")
            return dist.group.WORLD

        # EP composed with FSDP (`dp_shard`), context parallel (`cp`) and/or HSDP
        # (`dp_replicate`) on orthogonal mesh axes — read the ep group from accelerate's mesh, or
        # build one ourselves if accelerate hasn't (e.g., topology unit tests that drive
        # `_resolve_ep_group` directly). Experts shard on `ep` (tokens move via all-to-all); the
        # sequence shards on `cp` (DSA attention gathers the compressed KV on that axis);
        # non-expert weights shard on `dp_shard`. TP is still unsupported in composition.
        if tp_size > 1:
            raise NotImplementedError(
                "EP × TP composition not yet supported. Got "
                f"ep={ep_size}, dp_shard={dp_shard_size}, tp={tp_size}, cp={cp_size}. "
                "Supported: EP, EP × dp_shard, EP × cp, EP × cp × dp_shard."
            )
        from axolotl.monkeypatch.accelerate.parallelism_config import MESH_ORDER

        sizes = {
            "dp_replicate": dp_replicate_size,
            "dp_shard": dp_shard_size,
            "cp": cp_size,
            "ep": ep_size,
        }
        axes = [(n, sizes[n]) for n in MESH_ORDER if sizes.get(n, 1) > 1]
        mesh = ExpertParallelPlugin._accelerate_mesh()
        if mesh is None or "ep" not in (mesh.mesh_dim_names or ()):
            cached = ExpertParallelPlugin._device_mesh
            if (
                cached is not None
                and tuple(cached.mesh_dim_names or ()) == tuple(n for n, _ in axes)
                and tuple(cached.shape) == tuple(size for _, size in axes)
            ):
                # Before the trainer's Accelerator exists, reuse the mesh an earlier call built
                return cached["ep"].get_group()
            from torch.distributed.device_mesh import init_device_mesh

            # Fallback mesh in accelerate's axis order (ep innermost, node-local groups).
            mesh = init_device_mesh(
                "cuda" if torch.cuda.is_available() else "cpu",
                tuple(s for _, s in axes),
                mesh_dim_names=tuple(n for n, _ in axes),
            )
        ExpertParallelPlugin._device_mesh = mesh
        LOG.info(
            f"expert_parallel: ep mesh shape={tuple(mesh.shape)} "
            f"axes={mesh.mesh_dim_names}; ep group "
            f"members={dist.get_process_group_ranks(mesh['ep'].get_group())}"
        )
        return mesh["ep"].get_group()

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
        """Pre-wrap each Experts module (or the expert-LoRA ParamWrapper enclosing it) with
        FSDP on `dp_shard_mesh`: the non-ep data axes under composition, a per-rank size-1
        mesh under pure EP.

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

        root = dp_shard_mesh._get_root_mesh()
        ep_size = (
            root["ep"].size()
            if root is not None and "ep" in (root.mesh_dim_names or ())
            else 1
        )
        # Each rank's expert grad is the SUM over its ep-group's tokens (combine backward), while
        # dense grads are averaged over the whole world; divide the reduce-scatter by ep*dp_shard
        # so EP / FSDP / FSDP+EP produce the same effective gradient. A post-accumulate-grad hook
        # cannot do this: FSDP assigns sharded grads directly, so such hooks never fire.
        divide_factor = float(ep_size * dp_shard_mesh.size())

        # `target_parameters` expert LoRA lives on the ParamWrapper chain wrapping the experts
        # module (which `_detect_experts_modules` skips). Left to the outer decoder-layer auto-wrap
        # it shards on the FULL ep×dp mesh — i.e. ACROSS the ep axis — corrupting the per-ep-rank
        # expert slice (grads averaged over ranks owning different experts; save reconstructs the
        # wrong shape). Wrap the OUTERMOST expert ParamWrapper as its own FSDP unit on dp_shard so
        # FSDP unshards the adapter (incl. the nested inner wrapper's) to plain tensors right
        # before it is read. The experts module underneath must NOT be a separate unit: PEFT
        # registers its parametrization (baddbmm of W with the adapter) eagerly in the wrapper's
        # forward, before the experts forward would unshard W, so a nested unit leaves W a sharded
        # DTensor against plain adapter factors.
        all_pws = [m for _n, m in model.named_modules() if _is_param_wrapper(m)]
        inner = {getattr(pw, "base_layer", None) for pw in all_pws}
        outer_expert_pws = [
            pw
            for pw in all_pws
            if pw not in inner
            and _real_experts_base(pw) is not None
            and getattr(_real_experts_base(pw), "num_local_experts", None) is not None
        ]
        lora_wrapped_experts = {_real_experts_base(pw) for pw in outer_expert_pws}

        def _shard(module, **shard_kwargs):
            # FSDP2 moves CPU-resident wrapper-subclass params (packed NVFP4 experts an EP
            # scatter left on the CPU) by re-wrapping them in a fresh nn.Parameter, which
            # defaults to trainable; restore the frozen flag.
            frozen = {n for n, p in module.named_parameters() if not p.requires_grad}
            fully_shard(module, **shard_kwargs)
            for n, p in module.named_parameters():
                if n in frozen and p.requires_grad:
                    p.requires_grad_(False)

        wrapped = []
        for _name, module in _detect_experts_modules(model):
            if module in lora_wrapped_experts:
                continue
            _shard(module, **kwargs)
            wrapped.append(module)
        from .nvfp4_lora import (
            active as nvfp4_lora_active,
            expert_params,
            move_frozen_experts,
        )

        for pw in outer_expert_pws:
            pw_kwargs = kwargs
            if nvfp4_lora_active():
                # the frozen packed NVFP4 experts stay plain tensors on every rank: the
                # persistent kernels and the requant read their bytes in place, and an
                # all-gathered copy per forward would be freed under them. FSDP moves only
                # the params it manages, so the base goes to the device here.
                ignored = expert_params(_real_experts_base(pw))
                move_frozen_experts(_real_experts_base(pw), dp_shard_mesh.device_type)
                pw_kwargs = dict(kwargs, ignored_params=ignored)
            _shard(pw, **pw_kwargs)
            wrapped.append(pw)

        for module in wrapped:
            # A non-default factor makes FSDP reduce with NCCL PREMUL_SUM, which returns zeros
            # for bf16 (measured on H100, torch 2.13); plain SUM + a separate divide is exact.
            # A size-1 group never reduces and divides once itself: forcing SUM there divides twice.
            if dp_shard_mesh.size(-1) > 1:
                module.set_force_sum_reduction_for_comms(True)
            module.set_gradient_divide_factor(divide_factor)

        LOG.debug(
            f"expert_parallel: pre-wrapped Experts modules + {len(outer_expert_pws)} expert "
            f"ParamWrapper(s) on dp_shard mesh (size={dp_shard_mesh.size()}), "
            f"gradient divide factor {divide_factor}."
        )

    @staticmethod
    def _register_expert_grad_scale(model, ep_size: int) -> int:
        """Scale expert weight grads by `1/ep_size` so pure EP under DDP matches
        FSDP / FSDP+EP. Only for plain (non-FSDP) expert params: FSDP assigns
        sharded grads directly, so post-accumulate-grad hooks never fire there
        (`fully_shard_experts` sets the reduce-scatter divide factor instead).
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
    def _expert_layout(cfg, world_size: int) -> tuple[int, int, int, int, int]:
        """``(ep, dp_replicate, dp_shard, tp, cp)`` in the patched accelerate mesh's terms, where
        ``dp_shard`` is the expert FSDP split inside an EP group (``efsdp``)."""
        mesh = resolve_cfg_mesh(cfg, world_size)
        return mesh.ep, mesh.dp_replicate, mesh.efsdp, mesh.tp, mesh.cp

    @staticmethod
    def _is_ep_enabled(cfg) -> bool:
        """EP is enabled when expert_parallel_size > 1 (mirrors TP / DP UX)."""
        ep_size = getattr(cfg, "expert_parallel_size", 1) or 1
        return ep_size > 1

    @staticmethod
    def _reject_transformers_distribution(cfg) -> None:
        """transformers' own TP/EP (`distributed_config`, `tp_plan`) shards experts as DTensors
        under its RouterParallel plan; it cannot coexist with this plugin's sharding."""
        overrides = getattr(cfg, "overrides_of_model_kwargs", None) or {}
        clashing = sorted(
            k
            for k in ("distributed_config", "tp_plan", "tp_size", "device_mesh")
            if overrides.get(k) is not None
        )
        if clashing:
            raise ValueError(
                f"expert_parallel: model_kwargs {clashing} hand model sharding to transformers "
                "(its tp/ep plan), which conflicts with the expert_parallel plugin. Remove them; "
                "expert_parallel_size / dp_shard_size configure the mesh."
            )

    @staticmethod
    def _reject_transformers_distributed_model(model) -> None:
        config = getattr(model, "config", None)
        distributed_config = getattr(config, "distributed_config", None)
        if getattr(distributed_config, "enable_expert_parallel", False):
            raise ValueError(
                "expert_parallel: the model was loaded with transformers' "
                "`enable_expert_parallel=True`, so its experts are already sharded by "
                "transformers' ep plan. The expert_parallel plugin owns expert sharding; "
                "disable transformers' expert parallelism."
            )
        if (getattr(model, "_tp_size", None) or 1) > 1:
            raise ValueError(
                "expert_parallel: the model was loaded with transformers' tensor parallelism "
                f"(tp_size={model._tp_size}); EP x TP composition is not supported."
            )

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
        resolve_cfg_mesh(cfg, world_size)

    @classmethod
    def _resolve_backend(cls, cfg) -> str | None:
        """Resolve ``expert_parallel_backend`` and store it on cfg.

        ``auto`` picks DeepEP when importable, else torch (never ``optmoe``). Returns
        ``None`` when an explicit ``deep_ep`` is unavailable and the fallback is enabled (EP is
        then skipped). An unusable ``optmoe`` degrades to ``torch`` under the fallback."""
        backend = getattr(cfg, "expert_parallel_backend", None) or "auto"
        if backend == "auto":
            backend = "deep_ep" if find_spec("deep_ep") is not None else "torch"
            cfg.expert_parallel_backend = backend
        if backend == "deep_ep" and not cls._deep_ep_available(cfg):
            return None
        if backend == OPTMOE:
            reason = cls._optmoe_config_reason(cfg)
            if reason is not None:
                return cls._optmoe_fallback(cfg, reason)
        return backend

    @staticmethod
    def _deep_ep_available(cfg) -> bool:
        if find_spec("deep_ep") is not None:
            return True
        msg = (
            "expert_parallel_backend=deep_ep but `deep_ep` is not importable. "
            "See the integration README for install instructions."
        )
        if cfg.expert_parallel_fallback_on_unsupported:
            LOG.warning(msg + " Falling back to standard experts implementation.")
            return False
        raise ImportError(msg)
