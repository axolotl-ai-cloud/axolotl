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

    Axolotl folds plugin args into the full config by inheritance
    (``integrations/config.py::merge_input_args``), so the cross-field validator below runs on
    the merged config and sees the base-config fields (``load_in_4bit``, ``fsdp``, ...); the
    ``getattr`` defaults keep the model instantiable standalone as well.
    """

    expert_offload: bool = False
    """Keep frozen 4-bit MoE experts in CPU RAM and stream one block's experts to the GPU at a time,
    lowering peak VRAM so MoE models whose experts exceed VRAM can QLoRA-train on a single small GPU.
    Requires a 4-bit adapter (``load_in_4bit`` + ``adapter: qlora``), ``gradient_checkpointing: true``
    with ``use_reentrant: false``, and a single GPU (no FSDP / DeepSpeed / expert-parallel)."""

    expert_offload_pin_memory: bool = True
    """Home the offloaded expert weights in pinned CPU memory so the per-block host->device copy is
    truly asynchronous. Set false only if pinned memory is scarce (falls back to a correct but
    synchronous pageable copy)."""

    @model_validator(mode="after")
    def validate_expert_offload_requirements(self):
        if not self.expert_offload:
            return self

        errors: list[str] = []

        # 4-bit experts: the mechanism swaps ``Linear4bit.weight.data``.
        if not getattr(self, "load_in_4bit", False):
            errors.append(
                "requires load_in_4bit: true (it offloads 4-bit Linear4bit experts)"
            )
        if getattr(self, "adapter", None) not in ("lora", "qlora"):
            errors.append("requires adapter: qlora (or lora with load_in_4bit)")

        # Gradient checkpointing (use_reentrant=False) is load-bearing: the backward recompute
        # re-stages each block's experts, and it is what lets eviction actually free memory rather
        # than pin every staged weight alive as a matmul_4bit saved-for-backward tensor.
        # ``use_reentrant`` must be EXPLICITLY false: this validator runs at config-parse time,
        # before ``normalize_config`` defaults omitted kwargs to ``{"use_reentrant": True}``.
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

        # Single GPU: FSDP / DeepSpeed / expert-parallel move or shard these same weights and
        # would race the stage/evict swaps.
        if getattr(self, "fsdp_config", None) or getattr(self, "fsdp", None):
            errors.append("is single-GPU only and incompatible with FSDP")
        if getattr(self, "deepspeed", None):
            errors.append("is single-GPU only and incompatible with DeepSpeed")
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
