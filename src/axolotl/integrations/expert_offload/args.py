# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the expert-offload plugin."""

from pydantic import BaseModel


class ExpertOffloadArgs(BaseModel):
    """Input args for the expert_offload plugin. See the integration README.

    Cross-field requirements (4-bit adapter, gradient checkpointing, single-GPU) depend on config
    fields this model cannot see, so they are validated in the plugin's ``pre_model_load``.
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
