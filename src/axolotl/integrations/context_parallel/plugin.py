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

This is opt-in via the ``context_parallel:`` config block and is independent of
the legacy ``context_parallel_size`` ring-flash-attn path. Composition with FSDP2
uses the accelerate device mesh's ``cp`` dim when present; otherwise a standalone
CP mesh is built from the world group.
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
        self._hook_handles = []
        self._grad_accum = 1

    def get_input_args(self) -> str | None:
        return "axolotl.integrations.context_parallel.args.ContextParallelArgs"

    def register(self, cfg: dict):
        # Drive axolotl's native CP mesh from our nested config so accelerate builds
        # `cp` as a NON-DP mesh dim (FSDP2 reduces grads over it; DDP won't all-reduce
        # across CP ranks). The legacy ring-flash-attn path is bypassed (see train.py /
        # validation gates keyed on this plugin being present).
        cp = cfg.get("context_parallel") if isinstance(cfg, dict) else None
        size = (
            (cp or {}).get("size")
            if isinstance(cp, dict)
            else getattr(cp, "size", None)
        )
        if size and size > 1 and not cfg.get("context_parallel_size"):
            cfg["context_parallel_size"] = size

    @staticmethod
    def _cp_cfg(cfg):
        return getattr(cfg, "context_parallel", None)

    def _enabled(self, cfg) -> bool:
        cp = self._cp_cfg(cfg)
        return bool(cp and getattr(cp, "size", 1) and cp.size > 1)

    def pre_model_load(self, cfg):
        if not self._enabled(cfg):
            return
        from ringmaster.compat import require_torch

        require_torch()  # hard gate: torch >= 2.11

    def post_model_build(self, cfg, model):
        if not self._enabled(cfg):
            return

        import ringmaster as rm

        cp = self._cp_cfg(cfg)
        self._grad_accum = int(getattr(cfg, "gradient_accumulation_steps", 1) or 1)
        # GRPO/EBFT consume full-sequence logits, so the CP manager must gather the
        # sharded outputs back together (SFT computes loss per-shard and skips this).
        self._gather_outputs = str(getattr(cfg, "rl", None) or "").lower() in (
            "grpo",
            "ebft",
        )

        rm_cfg = rm.RingmasterConfig(
            size=cp.size,
            backend=rm.Backend(cp.backend),
            ulysses_size=cp.ulysses_size if cp.ulysses_size else rm.AUTO,
            ring_size=cp.ring_size if cp.ring_size else rm.AUTO,
            rotate_method=rm.RotateMethod(cp.rotate_method),
            load_balance=rm.LoadBalance(cp.load_balance),
            ring_impl=rm.RingImpl(cp.ring_impl),
        )

        inner_attn = self._resolve_inner_attn(cfg)
        num_kv_heads = self._num_kv_heads(model)
        device_mesh = self._accelerate_mesh()

        self._runtime = rm.setup(
            rm_cfg,
            num_kv_heads=num_kv_heads,
            device_mesh=device_mesh,
            cp_dim="cp",
            inner_attn=inner_attn,
        )

        LOG.info(
            "ringmaster CP enabled: size=%d backend=%s ulysses=%d ring=%d inner=%s",
            rm_cfg.size,
            rm_cfg.backend.value,
            rm_cfg.ulysses_size,
            rm_cfg.ring_size,
            inner_attn,
        )

        if self._runtime.attn_implementation:
            model.set_attn_implementation(self._runtime.attn_implementation)

        # Bridge ringmaster's CP group to axolotl's mamba-CP machinery (the nemotron_h /
        # falcon_h1 / granitemoehybrid patches force the unfused chunk-scan path and
        # apply the SSM state correction keyed on this group).
        try:
            from axolotl.monkeypatch.ring_attn import set_ring_attn_group

            set_ring_attn_group(self._runtime.cp_group)
        except Exception:  # pragma: no cover
            pass

        has_recurrent = self._wire_recurrent_layers(model)
        # Zigzag (head_tail) reorders the sequence non-contiguously, which breaks the
        # Mamba/GDN state-passing recurrence (it needs contiguous order). Downgrade it
        # to contiguous for hybrid models; distflash/none keep contiguous tokens so
        # they're SSM-safe and left as configured.
        if (
            has_recurrent
            and self._runtime.config.load_balance == rm.LoadBalance.HEAD_TAIL
        ):
            LOG.info(
                "ringmaster: hybrid SSM model — forcing contiguous CP sharding (zigzag permutes tokens, unsafe for the recurrence)"
            )
            self._runtime.config.load_balance = rm.LoadBalance.NONE

        # GLM-5.2 DSA owns its own CP attention (compressed-KV all-gather + per-rank
        # q_offset=rank*s_local); ringmaster only shards the batch for it. That kernel
        # hard-assumes each rank owns a contiguous span [r*L, (r+1)*L), so any zigzag
        # shard (e.g. an explicit backend: ring) would silently corrupt it. Force
        # contiguous regardless of the resolved backend.
        if (
            self._glm_dsa_requires_contiguous(cfg)
            and self._runtime.config.load_balance == rm.LoadBalance.HEAD_TAIL
        ):
            LOG.info(
                "ringmaster: use_glm_dsa_kernels — forcing contiguous CP sharding (DSA attention requires contiguous per-rank spans)"
            )
            self._runtime.config.load_balance = rm.LoadBalance.NONE

        self._install_hooks(model)

    @staticmethod
    def _glm_dsa_requires_contiguous(cfg) -> bool:
        """GLM-5.2 DSA owns its own CP attention and needs contiguous per-rank spans
        (``q_offset=rank*s_local``), so ringmaster must not zigzag-shard for it."""
        return bool(getattr(cfg, "use_glm_dsa_kernels", False))

    @staticmethod
    def _wire_recurrent_layers(model):
        """Apply Mamba2/linear-attention CP state-passing to recurrent mixer layers
        (Nemotron-H, Qwen3.5/Next, Falcon-H1, Granite-MoE-Hybrid). Delegates to
        ringmaster's shared helper so the plugin and the trl/raw examples stay in
        sync; no-op without the mamba-ssm kernel. Returns True if anything was wired.
        """
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
        try:
            import ringmaster as rm

            rm.teardown()
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _resolve_inner_attn(cfg) -> str:
        impl = getattr(cfg, "attn_implementation", None)
        if impl and impl not in ("eager", "sdpa"):
            return impl
        if getattr(cfg, "flash_attention", False):
            return "flash_attention_2"
        if impl:
            return impl
        return "flash_attention_2"

    @staticmethod
    def _num_kv_heads(model) -> int | None:
        config = getattr(model, "config", None)
        if config is None:
            return None
        text_config = getattr(config, "text_config", config)  # multimodal
        return getattr(text_config, "num_key_value_heads", None) or getattr(
            text_config, "num_attention_heads", None
        )

    @staticmethod
    def _accelerate_mesh():
        try:
            from accelerate.state import AcceleratorState

            state = AcceleratorState()
            mesh = getattr(state, "torch_device_mesh", None) or getattr(
                state, "device_mesh", None
            )
            return mesh
        except Exception:  # pragma: no cover - mesh may not exist (pure CP)
            return None

    def _install_hooks(self, model):
        # The per-step CP data orchestration (shape-safe broadcast → contiguous
        # shard → num_items normalization → eval-loss correction → output gather)
        # lives in ringmaster so axolotl, trl and standalone scripts share one
        # implementation. This plugin only wires it to the model + CP group.
        from ringmaster import ContextParallelContextManager

        self._cp_ctx = ContextParallelContextManager(
            [model],
            self._runtime.cp_group,
            gradient_accumulation_steps=self._grad_accum,
            gather_outputs=getattr(self, "_gather_outputs", False),
            # shard_load_balance is the single source of truth: zigzag only for pure
            # ring; Ulysses/USP/distflash/none stay contiguous (Ulysses/USP gather the
            # full sequence, so a zigzag shard would scramble the gathered order).
            load_balance=self._runtime.shard_load_balance,
        )
        self._hook_handles.extend(self._cp_ctx.install())
