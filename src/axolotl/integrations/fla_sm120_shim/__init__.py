# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Apply the ringmaster sm_120 fla TileLang warp-spec shim.

On cu13 + sm_120 (Blackwell consumer), flash-linear-attention's gated-delta
backward dispatches to a TileLang kernel whose warp-specialized MMA pass emits
misaligned shared-memory descriptors (CUDA_ERROR_MISALIGNED_ADDRESS, fla #913).
This affects Qwen3.5 / Qwen3-Next training whether or not context parallelism is
used. The shim disables only that pass (numerics unchanged). The ContextParallel
plugin already applies it; load this plugin for non-CP runs.
"""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class FlaSm120ShimPlugin(BasePlugin):
    def get_input_args(self) -> str | None:
        return None

    def pre_model_load(self, cfg):
        try:
            from ringmaster.strategies.linear_attn import _apply_fla_sm120_shim

            if _apply_fla_sm120_shim():
                LOG.info("Applied fla sm_120 TileLang warp-spec shim")
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("fla sm_120 shim not applied: %s", exc)
        try:
            from ringmaster.strategies.mamba import (
                ensure_causal_conv1d_cuda_export,
                prefer_local_mamba_kernels,
            )

            if prefer_local_mamba_kernels():
                LOG.info("Preset local mamba-ssm/causal-conv1d kernels (skip kernels-hub)")
            ensure_causal_conv1d_cuda_export()
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("mamba kernel preset not applied: %s", exc)

    def post_model_build(self, cfg, model):
        import os

        # Opt-in: route hybrid-SSM mixers off the fused mem-efficient path (which
        # calls the raw causal_conv1d_cuda.causal_conv1d_fwd ABI) onto the unfused
        # causal_conv1d_fn wrapper. Needed when the installed causal_conv1d major
        # version mismatches the kernels-hub mamba build (CP already forces this).
        if os.environ.get("RINGMASTER_FORCE_UNFUSED_MAMBA") != "1":
            return
        n = 0
        for module in model.modules():
            if getattr(module, "use_mem_eff_path", None) is True:
                module.use_mem_eff_path = False
                n += 1
        if n:
            LOG.info("Forced use_mem_eff_path=False on %d SSM mixers", n)
