# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the expert-offload plugin."""

from pydantic import BaseModel, model_validator


class ExpertOffloadArgs(BaseModel):
    """Input args for the expert_offload plugin. See the integration README.

    ``merge_input_args`` folds plugin args into the full config, so the validator below sees the
    merged base-config fields; the ``getattr`` defaults keep the model instantiable standalone.
    """

    expert_offload: bool = False
    """Stream frozen 4-bit MoE experts from pinned CPU RAM to the GPU one block at a time, cutting
    peak VRAM. Requires ``load_in_4bit`` + ``adapter: qlora`` and non-reentrant gradient
    checkpointing; single-GPU or plain DDP. See the integration README."""

    expert_offload_pin_memory: bool = True
    """Home offloaded experts in pinned CPU memory for async H2D copies; set false only if pinned
    memory is scarce."""

    @model_validator(mode="after")
    def validate_expert_offload_requirements(self):
        if not self.expert_offload:
            return self

        errors: list[str] = []

        # The mechanism swaps packed 4-bit ``weight.data`` — 4-bit only.
        if not getattr(self, "load_in_4bit", False):
            errors.append(
                "requires load_in_4bit: true (it offloads 4-bit Linear4bit experts)"
            )
        if getattr(self, "adapter", None) not in ("lora", "qlora"):
            errors.append("requires adapter: qlora (or lora with load_in_4bit)")

        # The recompute is what makes eviction correct and memory-freeing. ``use_reentrant`` must
        # be EXPLICITLY false: this runs before ``normalize_config`` defaults it to true.
        if not getattr(self, "gradient_checkpointing", False):
            errors.append(
                "requires gradient_checkpointing: true (the backward recompute re-stages experts; "
                "without it, offload is neither correct nor memory-saving)"
            )
        elif (getattr(self, "gradient_checkpointing_kwargs", None) or {}).get(
            "use_reentrant"
        ) is not False:
            errors.append(
                "requires an explicit gradient_checkpointing_kwargs.use_reentrant: false "
                "(axolotl defaults it to true when omitted; reentrant checkpointing does not "
                "re-run the block pre-hook on recompute)"
            )

        # FSDP / DeepSpeed / expert-parallel move or shard the same weights (plain DDP is fine).
        if getattr(self, "fsdp_config", None) or getattr(self, "fsdp", None):
            errors.append("is incompatible with FSDP (use single-GPU or plain DDP)")
        if getattr(self, "deepspeed", None):
            errors.append(
                "is incompatible with DeepSpeed (use single-GPU or plain DDP)"
            )
        if (getattr(self, "expert_parallel_size", None) or 1) > 1:
            errors.append(
                "is incompatible with expert_parallel (both manage the expert weights)"
            )

        if errors:
            raise ValueError(
                "expert_offload is enabled but the config is incompatible:\n  - "
                + "\n  - ".join(errors)
            )
        return self
