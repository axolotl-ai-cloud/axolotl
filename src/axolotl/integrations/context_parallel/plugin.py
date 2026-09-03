# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Context-parallel plugin backed by the standalone ``ringmaster`` package.

Phase 1 wires the Ulysses backend: it switches the model's attention to a
ringmaster wrapper (all-to-all around the model's own HF kernel) and installs a
forward pre-hook that shards each batch along the sequence dim, plus loss
corrections across the CP group.

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

    def get_input_args(self) -> str | None:
        return "axolotl.integrations.context_parallel.args.ContextParallelArgs"

    def register(self, cfg: dict):
        # cfg is the raw pre-validation dict; keep `context_parallel.size` and the
        # flat `context_parallel_size` (which drives accelerate's cp mesh dim) in sync.
        size = (cfg.get("context_parallel") or {}).get("size")
        flat = cfg.get("context_parallel_size")
        if size and size > 1:
            if flat and flat != size:
                raise ValueError(
                    f"context_parallel.size ({size}) conflicts with "
                    f"context_parallel_size ({flat}); set only one"
                )
            cfg["context_parallel_size"] = size
        elif flat and flat > 1 and not cfg.get("context_parallel"):
            cfg["context_parallel"] = {"size": flat}

    @staticmethod
    def _cp_cfg(cfg):
        return getattr(cfg, "context_parallel", None)

    def _enabled(self, cfg) -> bool:
        cp = self._cp_cfg(cfg)
        return bool(cp and getattr(cp, "size", 1) and cp.size > 1)

    def pre_model_load(self, cfg):
        if not self._enabled(cfg):
            return
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
            load_balance=rm.LoadBalance(cp.load_balance),
            ring_impl=rm.RingImpl(cp.ring_impl),
        )

        models = [trainer.model]
        ref_model = getattr(trainer, "ref_model", None)
        if ref_model is not None:
            models.append(ref_model)

        inner_attn = self._resolve_inner_attn(cfg)
        num_kv_heads = self._num_kv_heads(models[0])
        device_mesh = getattr(trainer.accelerator, "torch_device_mesh", None)

        self._runtime = rm.setup(
            rm_cfg,
            num_kv_heads=num_kv_heads,
            device_mesh=device_mesh,
            cp_dim="cp",
            inner_attn=inner_attn,
        )

        LOG.info(
            "ringmaster CP enabled: size=%d backend=%s ulysses=%d ring=%d inner=%s mesh=%s",
            rm_cfg.size,
            rm_cfg.backend.value,
            rm_cfg.ulysses_size,
            rm_cfg.ring_size,
            inner_attn,
            "accelerate" if device_mesh is not None else "standalone",
        )

        glm_dsa = self._glm_dsa_requires_contiguous(cfg)
        if self._runtime.attn_implementation and not glm_dsa:
            for model in models:
                model.set_attn_implementation(self._runtime.attn_implementation)

        # The mamba/SSM CP corrections and the GRPO trainer resolve the CP group
        # through this registry.
        from axolotl.monkeypatch.ring_attn import set_ring_attn_group

        set_ring_attn_group(self._runtime.cp_group)

        has_recurrent = any(self._wire_recurrent_layers(model) for model in models)

        # Zigzag sharding permutes tokens; anything that needs contiguous per-rank
        # spans must downgrade to LoadBalance.NONE (contiguous, always correct).
        contiguity_reason = None
        if has_recurrent:
            contiguity_reason = "hybrid SSM recurrence needs contiguous token order"
        elif glm_dsa:
            contiguity_reason = "GLM DSA attention assumes q_offset=rank*s_local"
        elif self._gather_outputs:
            contiguity_reason = "output gathering reassembles contiguous shards"
        if (
            contiguity_reason
            and self._runtime.config.load_balance == rm.LoadBalance.HEAD_TAIL
        ):
            LOG.info(
                "ringmaster: forcing contiguous CP sharding (%s)", contiguity_reason
            )
            self._runtime.config.load_balance = rm.LoadBalance.NONE

        self._install_hooks(models, cfg)

    @staticmethod
    def _glm_dsa_requires_contiguous(cfg) -> bool:
        """GLM-5.2 DSA owns its own CP attention and needs contiguous per-rank spans
        (``q_offset=rank*s_local``), so ringmaster must not zigzag-shard for it."""
        return bool(getattr(cfg, "use_glm_dsa_kernels", False))

    @staticmethod
    def _wire_recurrent_layers(model):
        """Apply Mamba2/linear-attention CP state-passing to recurrent mixer layers.
        No-op without the mamba-ssm kernel. Returns True if anything was wired."""
        import ringmaster as rm

        wiring = rm.wire_recurrent_layers(model)
        if wiring:
            LOG.info(
                "ringmaster: wired recurrent-layer CP (mamba modules=%s, linear-attn mixers=%d)",
                list(wiring.mamba_modules),
                wiring.linear_attn_mixers,
            )
        return bool(wiring)

    def post_train_unload(self, cfg):
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
        from ringmaster import ContextParallelContextManager

        grad_accum = int(getattr(cfg, "gradient_accumulation_steps", 1) or 1)
        if self._gather_outputs:
            for model in models:
                self._hook_handles.append(
                    model.register_forward_pre_hook(
                        self._strip_logits_to_keep_pre_hook, with_kwargs=True
                    )
                )

        self._cp_ctx = ContextParallelContextManager(
            models,
            self._runtime.cp_group,
            gradient_accumulation_steps=grad_accum,
            gather_outputs=self._gather_outputs,
            # zigzag only for pure ring; Ulysses/USP gather the full sequence, so a
            # zigzag shard would scramble the gathered order.
            load_balance=self._runtime.shard_load_balance,
        )
        self._hook_handles.extend(self._cp_ctx.install())
