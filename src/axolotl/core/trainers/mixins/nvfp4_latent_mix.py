"""Trainer mixin applying ``nvfp4_merge_aware_latent_mix`` per training micro-batch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import Trainer

if TYPE_CHECKING:
    from axolotl.integrations.kernels.merge_aware_latent_mix import NVFP4LatentMix


class NVFP4LatentMixMixin(Trainer):
    """Hold one latent-mix draw across each ``training_step`` (forward and backward)."""

    _nvfp4_latent_mix: NVFP4LatentMix | None = None
    _nvfp4_latent_mix_ready = False

    def _get_nvfp4_latent_mix(self) -> NVFP4LatentMix | None:
        if not self._nvfp4_latent_mix_ready:
            self._nvfp4_latent_mix_ready = True
            cfg = getattr(self, "axolotl_cfg", None)
            if cfg is not None and cfg.get("nvfp4_merge_aware_latent_mix"):
                from axolotl.integrations.kernels.merge_aware_latent_mix import (
                    build_latent_mix,
                )

                self._nvfp4_latent_mix = build_latent_mix(cfg, self.model, self.args)
        return self._nvfp4_latent_mix

    def training_step(self, *args, **kwargs):
        latent_mix = self._get_nvfp4_latent_mix()
        if latent_mix is None:
            return super().training_step(*args, **kwargs)
        with latent_mix.micro_batch():
            return super().training_step(*args, **kwargs)
