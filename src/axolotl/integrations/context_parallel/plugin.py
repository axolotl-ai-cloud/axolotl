# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Context-parallel plugin backed by the standalone ``ringmaster`` package.

Switches attention to a ringmaster wrapper and installs batch-sharding hooks
while preserving Trainer's accumulation-window loss normalization.

Setup runs in ``post_trainer_create`` because the accelerate device mesh (the
``cp`` dim FSDP2 reduces gradients over) only exists once the trainer's
Accelerator has been constructed.
"""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ContextParallelPlugin(BasePlugin):
    """Long-context attention (Ulysses/Ring/USP) via ringmaster."""

    def __init__(self):
        super().__init__()
        self._runtime = None
        self._cp_ctx = None
        self._hook_handles = []
        self._gather_outputs = False
        self._restore_trainer = None
        self._restore_recurrent = None
        self._original_caches = []
        self._attention_configs = []

    def get_input_args(self) -> str | None:
        return "axolotl.integrations.context_parallel.args.ContextParallelArgs"

    def register(self, cfg: dict):
        # cfg is the raw pre-validation dict; keep `context_parallel.size` and the
        # flat `context_parallel_size` (which drives accelerate's cp mesh dim) in sync.
        block = cfg.get("context_parallel") or {}
        size = block.get("size")
        flat = cfg.get("context_parallel_size")
        if flat is None:
            flat = cfg.get("sequence_parallel_degree")
        if size is not None and flat is not None and flat != size:
            raise ValueError(
                f"context_parallel.size ({size}) conflicts with "
                f"context_parallel_size ({flat}); set only one"
            )
        if size is not None:
            cfg["context_parallel_size"] = size
        elif flat is not None:
            cfg["context_parallel_size"] = flat
            cfg["context_parallel"] = {**block, "size": flat}

    @staticmethod
    def _cp_cfg(cfg):
        return getattr(cfg, "context_parallel", None)

    def _enabled(self, cfg) -> bool:
        cp = self._cp_cfg(cfg)
        return bool(cp and getattr(cp, "size", 1) and cp.size > 1)

    def pre_model_load(self, cfg):
        if not self._enabled(cfg):
            return
        from .settings import check_model_capability

        check_model_capability(getattr(cfg, "model_config_type", None))
        try:
            from ringmaster.compat import require_torch
        except ImportError as exception:
            raise ImportError(
                "context parallelism requires the ringmaster package; install it "
                "with `pip install axolotl[ringmaster]` or "
                "`pip install axolotl-ringmaster`"
            ) from exception

        require_torch()  # hard gate: torch >= 2.11

    def post_trainer_create(self, cfg, trainer):
        if not self._enabled(cfg):
            return
        try:
            self._configure(cfg, trainer)
        except Exception:
            if self._runtime is not None:
                self.post_train_unload(cfg)
            raise

    def _configure(self, cfg, trainer):
        import ringmaster as rm

        cp = self._cp_cfg(cfg)
        from axolotl.utils.schemas.enums import RLType

        rl = getattr(cfg, "rl", None)
        # GRPO/EBFT consume full-sequence logits, so the CP manager must gather the
        # sharded outputs back together (SFT computes loss per-shard and skips this).
        self._gather_outputs = rl in (RLType.GRPO, RLType.EBFT)

        rm_cfg = rm.RingmasterConfig(
            size=cp.size,
            backend=rm.Backend(cp.backend),
            ulysses_size=cp.ulysses_size if cp.ulysses_size else rm.AUTO,
            ring_size=cp.ring_size if cp.ring_size else rm.AUTO,
            rotate_method=rm.RotateMethod(cp.rotate_method),
            load_balance=rm.LoadBalance.NONE,
            ring_impl=rm.RingImpl(cp.ring_impl),
        )

        models = [trainer.model]
        ref_model = getattr(trainer, "ref_model", None)
        if ref_model is not None:
            models.append(ref_model)

        inner_attn = self._resolve_inner_attn(cfg)
        num_kv_heads = self._num_kv_heads(models[0])
        device_mesh = getattr(trainer.accelerator, "torch_device_mesh", None)

        from ringmaster.mesh import intra_node_size

        rm_cfg.normalize(num_kv_heads=num_kv_heads, intra_node_size=intra_node_size())
        from ringmaster.strategies.state_passing import recurrent_plan

        from .settings import check_model_capability, resolve_settings

        for model in models:
            model_config = model.config
            check_model_capability(getattr(model_config, "model_type", None))
            if hasattr(model_config, "get_text_config"):
                check_model_capability(
                    getattr(model_config.get_text_config(), "model_type", None)
                )

        mixers, kda_mixers, mamba_mixers = recurrent_plan(models, cp.size)
        if (mixers or kda_mixers) and (getattr(cfg, "micro_batch_size", 1) or 1) != 1:
            raise ValueError("FLA context parallelism requires micro_batch_size: 1")
        recurrent = bool(mixers or kda_mixers or mamba_mixers)
        glm_dsa = self._glm_dsa_requires_contiguous(cfg)
        reason = (
            "recurrent state passing"
            if recurrent
            else "GLM DSA attention"
            if glm_dsa
            else "output gathering"
            if self._gather_outputs
            else None
        )
        communication = resolve_settings(
            cp,
            rm_cfg,
            num_kv_heads=num_kv_heads,
            contiguous_reason=reason,
            sliding_window=any(
                getattr(module, "sliding_window", None)
                for model in models
                for module in model.modules()
            ),
            glm_dsa=glm_dsa,
            inner_attn=inner_attn,
            dropout=max(
                float(
                    getattr(
                        model.config.get_text_config()
                        if hasattr(model.config, "get_text_config")
                        else model.config,
                        "attention_dropout",
                        0.0,
                    )
                    or 0.0
                )
                for model in models
            ),
        )
        if rm_cfg.ring_size > 1 and not glm_dsa:
            from ringmaster.strategies.ring import resolve_ring_impl

            if (
                resolve_ring_impl(rm_cfg.ring_impl, inner_attn)
                == rm.RingImpl.TORCH_NATIVE
            ):
                raise ValueError(
                    "Ringmaster's torch_native Ring kernel is forward-only. "
                    "For Ring/USP training use flash_attention_2/3/4 with "
                    "ring_impl: hf_kernels, or use backend: ulysses with SDPA."
                )
        if device_mesh is not None and rm_cfg.ulysses_size > 1 and rm_cfg.ring_size > 1:
            from .mesh import RingmasterMesh

            device_mesh = RingmasterMesh(
                device_mesh,
                ring_size=rm_cfg.ring_size,
                ulysses_size=rm_cfg.ulysses_size,
            )

        self._runtime = rm.setup(
            rm_cfg,
            num_kv_heads=num_kv_heads,
            device_mesh=device_mesh,
            cp_dim="cp",
            inner_attn=inner_attn,
        )

        LOG.info(
            "ringmaster CP enabled: size=%d backend=%s ulysses=%d ring=%d inner=%s mesh=%s balance=%s communication=%s recurrent=%s",
            rm_cfg.size,
            rm_cfg.backend.value,
            rm_cfg.ulysses_size,
            rm_cfg.ring_size,
            inner_attn,
            "accelerate" if device_mesh is not None else "standalone",
            rm_cfg.load_balance.value,
            communication,
            "fla_native" if mixers or kda_mixers else "mamba" if recurrent else "none",
        )

        if self._runtime.attn_implementation and not glm_dsa:
            seen_configs = set()
            for model in models:
                for module in [model, *model.modules()]:
                    config = getattr(module, "config", None)
                    if config is not None and id(config) not in seen_configs:
                        seen_configs.add(id(config))
                        if hasattr(config, "_attn_implementation"):
                            self._attention_configs.append(
                                (config, config._attn_implementation)
                            )
                model.set_attn_implementation(self._runtime.attn_implementation)

        # The mamba/SSM CP corrections and the GRPO trainer resolve the CP group
        # through this registry.
        from axolotl.monkeypatch.ring_attn import set_ring_attn_group

        set_ring_attn_group(self._runtime.cp_group)

        if recurrent:
            from ringmaster import wire_recurrent_layers

            restores = []
            self._restore_recurrent = lambda: [
                restore() for restore in reversed(restores)
            ]
            for model in models:
                restores.append(
                    wire_recurrent_layers(model, group=self._runtime.cp_group).restore
                )

        if recurrent:
            for model in models:
                config = (
                    model.config.get_text_config()
                    if hasattr(model.config, "get_text_config")
                    else model.config
                )
                self._original_caches.append(
                    (
                        config,
                        hasattr(config, "use_cache"),
                        getattr(config, "use_cache", None),
                    )
                )
                config.use_cache = False

        from .trainer import configure_trainer

        self._restore_trainer = configure_trainer(
            trainer, gather_outputs=self._gather_outputs
        )
        self._install_hooks(models, cfg)

    @staticmethod
    def _glm_dsa_requires_contiguous(cfg) -> bool:
        """GLM-5.2 DSA owns its own CP attention and needs contiguous per-rank spans
        (``q_offset=rank*s_local``), so ringmaster must not zigzag-shard for it."""
        return bool(getattr(cfg, "use_glm_dsa_kernels", False))

    def post_train_unload(self, cfg):
        for config, implementation in self._attention_configs:
            config._attn_implementation = implementation
        self._attention_configs = []
        for config, existed, value in self._original_caches:
            if existed:
                config.use_cache = value
            else:
                del config.use_cache
        self._original_caches = []
        if self._restore_recurrent is not None:
            self._restore_recurrent()
            self._restore_recurrent = None
        if self._restore_trainer is not None:
            self._restore_trainer()
            self._restore_trainer = None
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._cp_ctx = None
        self._runtime = None
        try:
            from axolotl.monkeypatch.ring_attn import set_ring_attn_group

            set_ring_attn_group(None)
            import ringmaster as rm

            rm.teardown()
        except ImportError:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _resolve_inner_attn(cfg) -> str:
        # Config validation canonicalizes legacy flags into attn_implementation;
        # the flash_attention fallback covers raw/unvalidated cfgs. Default to sdpa
        # rather than silently forcing a flash kernel.
        impl = getattr(cfg, "attn_implementation", None)
        if impl:
            return impl
        if getattr(cfg, "flash_attention", False):
            return "flash_attention_2"
        return "sdpa"

    @staticmethod
    def _num_kv_heads(model) -> int | None:
        config = getattr(model, "config", None)
        if config is None:
            return None
        if hasattr(config, "get_text_config"):
            text_config = config.get_text_config()
        else:
            text_config = getattr(config, "text_config", config)
        return getattr(text_config, "num_key_value_heads", None) or getattr(
            text_config, "num_attention_heads", None
        )

    @staticmethod
    def _strip_logits_to_keep_pre_hook(module, args, kwargs):
        # ringmaster has no varlen logits_to_keep support: an integer N would make
        # each rank keep the last N positions of its own shard (wrong global
        # positions). Dropping it computes full local logits; the gathered output is
        # re-sliced by the trainer (TRL slices logits[:, -logits_to_keep:]).
        for key in ("logits_to_keep", "num_logits_to_keep"):
            value = kwargs.get(key)
            if isinstance(value, int) and value:
                kwargs.pop(key)
        return args, kwargs

    def _install_hooks(self, models, cfg):
        from .trainer import TrainerContextParallelContextManager

        grad_accum = int(getattr(cfg, "gradient_accumulation_steps", 1) or 1)
        if self._gather_outputs:
            for model in models:
                self._hook_handles.append(
                    model.register_forward_pre_hook(
                        self._strip_logits_to_keep_pre_hook, with_kwargs=True
                    )
                )

        self._cp_ctx = TrainerContextParallelContextManager(
            models,
            self._runtime.cp_group,
            gradient_accumulation_steps=grad_accum,
            gather_outputs=self._gather_outputs,
            # zigzag only for pure ring; Ulysses/USP gather the full sequence, so a
            # zigzag shard would scramble the gathered order.
            load_balance=self._runtime.shard_load_balance,
        )
        self._hook_handles.extend(self._cp_ctx.install())
