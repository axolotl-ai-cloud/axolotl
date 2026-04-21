# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

from .lora_layout import (
    peft_down_proj_lora_to_scattermoe,
    peft_lora_B_to_scattermoe,
    peft_lora_to_scattermoe,
    validate_scattermoe_lora_shapes,
)

__all__ = [
    "peft_down_proj_lora_to_scattermoe",
    "peft_lora_B_to_scattermoe",
    "peft_lora_to_scattermoe",
    "validate_scattermoe_lora_shapes",
]

try:
    from . import layers
    from .lora_ops import ParallelExperts
    from .parallel_experts import flatten_sort_count, parallel_linear
    from .parallel_linear_lora import ScatterMoELoRA, parallel_linear_lora
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
else:
    __all__ += [
        "layers",
        "ParallelExperts",
        "flatten_sort_count",
        "parallel_linear",
        "ScatterMoELoRA",
        "parallel_linear_lora",
        "lora_ops",
    ]
