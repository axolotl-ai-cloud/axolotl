# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Latent mixing for merge-aware NVFP4 LoRA (``nvfp4_merge_aware_latent_mix``).

Merge-aware training optimizes ``Q(dequant(W) + s * B @ A)``, the merged
checkpoint, and never evaluates the unmerged ``dequant(W) + s * B @ A`` that
runtime-LoRA serving computes. With probability ``p`` per training micro-batch,
the sampler routes the merge-aware projections through PEFT's ordinary LoRA
forward instead, so both functions are trained. The choice is held for the
whole micro-batch (forward and backward) so gradient-checkpoint recomputation
replays the same forward.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

NATIVE = "native"
SONICMOE = "sonicmoe"


def mark_latent_mix_path(model: torch.nn.Module, path: str) -> None:
    """Record that ``model`` trains through a merge-aware path the sampler can toggle."""
    paths = set(getattr(model, "_axolotl_nvfp4_latent_mix_paths", ()))
    paths.add(path)
    model._axolotl_nvfp4_latent_mix_paths = frozenset(paths)


def latent_mix_paths(model: torch.nn.Module | None) -> frozenset[str]:
    return frozenset(getattr(model, "_axolotl_nvfp4_latent_mix_paths", ()))


class NVFP4LatentMix:
    """Per-micro-batch Bernoulli(``p``) choice of the unmerged LoRA forward.

    Draws come from a private RNG seeded by ``(seed, process_index)``, so the
    global RNG (data order, dropout) is untouched and data-parallel ranks draw
    independently.
    """

    def __init__(
        self,
        p: float,
        paths: frozenset[str],
        seed: int = 0,
        process_index: int = 0,
    ):
        if not 0 <= p < 1:
            raise ValueError(f"latent mix probability must be in [0, 1), got {p!r}")
        self.p = p
        self.paths = paths
        # training-time sampling, not security-sensitive
        self._rng = random.Random(f"nvfp4-latent-mix:{seed}:{process_index}")  # nosec B311
        self.micro_batches = 0
        self.latent_micro_batches = 0

    @contextlib.contextmanager
    def micro_batch(self) -> Iterator[bool]:
        """Draw for one micro-batch; yields True when it runs the unmerged forward."""
        latent = self._rng.random() < self.p
        self.micro_batches += 1
        self.latent_micro_batches += int(latent)
        if not latent:
            yield False
            return

        native = NATIVE in self.paths
        sonicmoe = False
        if native:
            from axolotl.monkeypatch.torchao_nvfp4_merge import (
                set_native_latent_forward,
            )

            set_native_latent_forward(True)
        if SONICMOE in self.paths:
            from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
                merge_aware_enabled,
                set_merge_aware_enabled,
            )

            # Before start_step the flag is still off and there is nothing to toggle.
            sonicmoe = merge_aware_enabled()
            if sonicmoe:
                set_merge_aware_enabled(False)
        try:
            yield True
        finally:
            if native:
                set_native_latent_forward(False)
            if sonicmoe:
                set_merge_aware_enabled(True)


def build_latent_mix(cfg, model, args) -> NVFP4LatentMix | None:
    """Return a sampler when ``nvfp4_merge_aware_latent_mix`` is set and supported."""
    p = cfg.get("nvfp4_merge_aware_latent_mix") if cfg is not None else None
    if not p:
        return None
    paths = latent_mix_paths(model)
    if not paths:
        LOG.warning(
            "nvfp4_merge_aware_latent_mix=%s has no effect: no merge-aware NVFP4 LoRA "
            "path that supports it was installed for this run (supported: the "
            "unsharded native TorchAO path and SonicMoE).",
            p,
        )
        return None
    LOG.info(
        "NVFP4 latent mix: %.0f%% of training micro-batches use the unmerged LoRA "
        "forward (%s)",
        100 * p,
        ", ".join(sorted(paths)),
    )
    return NVFP4LatentMix(
        p,
        paths,
        seed=int(args.seed),
        process_index=int(args.process_index),
    )
