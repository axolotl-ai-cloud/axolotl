# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-offload plugin for axolotl: single-GPU CPU offload of frozen 4-bit MoE experts."""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ExpertOffloadPlugin(BasePlugin):
    """Stream frozen 4-bit MoE experts from pinned CPU RAM one block at a time.

    Surgical counterpart to ``layer_offloading``: it moves only the frozen 4-bit experts (the bulk
    of a MoE's parameters) while attention, router, norms and the trainable LoRA adapters stay
    GPU-resident — minimizing per-step PCIe traffic for the same peak-VRAM reduction. Single-GPU
    QLoRA only. See ``offload.py`` for the mechanism and this integration's README.
    """

    def get_input_args(self):
        return "axolotl.integrations.expert_offload.ExpertOffloadArgs"

    def pre_model_load(self, cfg):
        """Validate the cross-field requirements the args model can't see, before the model loads."""
        if not getattr(cfg, "expert_offload", False):
            return
        self._validate(cfg)

    @staticmethod
    def _validate(cfg) -> None:
        errors: list[str] = []

        # 4-bit experts: the mechanism swaps ``Linear4bit.weight.data``.
        if not getattr(cfg, "load_in_4bit", False):
            errors.append(
                "requires load_in_4bit: true (it offloads 4-bit Linear4bit experts)"
            )
        if getattr(cfg, "adapter", None) not in ("lora", "qlora"):
            errors.append("requires adapter: qlora (or lora with load_in_4bit)")

        # Gradient checkpointing (use_reentrant=False) is load-bearing: the backward recompute
        # re-stages each block's experts, and it is what lets eviction actually free memory rather
        # than pin every staged weight alive as a matmul_4bit saved-for-backward tensor.
        if not getattr(cfg, "gradient_checkpointing", False):
            errors.append(
                "requires gradient_checkpointing: true (the backward recompute re-stages experts; "
                "without it, offload is neither correct nor memory-saving)"
            )
        else:
            use_reentrant = (
                getattr(cfg, "gradient_checkpointing_kwargs", None) or {}
            ).get("use_reentrant", False)
            if use_reentrant:
                errors.append(
                    "requires gradient_checkpointing_kwargs.use_reentrant: false "
                    "(reentrant checkpointing does not re-run the block pre-hook on recompute)"
                )

        # Single GPU: FSDP / DeepSpeed / expert-parallel move or shard these same weights and would
        # race the stage/evict swaps.
        if getattr(cfg, "fsdp_config", None) or getattr(cfg, "fsdp", None):
            errors.append("is single-GPU only and incompatible with FSDP")
        if getattr(cfg, "deepspeed", None):
            errors.append("is single-GPU only and incompatible with DeepSpeed")
        if (getattr(cfg, "expert_parallel_size", None) or 1) > 1:
            errors.append(
                "is incompatible with expert_parallel (both manage the expert weights)"
            )

        if errors:
            raise ValueError(
                "expert_offload is enabled but the config is incompatible:\n  - "
                + "\n  - ".join(errors)
            )

    def post_model_load(self, cfg, model):
        """Install the offload after the model is built, quantized, PEFT-wrapped and on the GPU."""
        if not getattr(cfg, "expert_offload", False):
            return
        from .offload import install_expert_offload

        install_expert_offload(
            model, pin=getattr(cfg, "expert_offload_pin_memory", True)
        )
