# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

# Local kernel loading must not register a second set of torch operators.
from axolotl.integrations.kernels.libs.scattermoe_lora import layers, lora_ops
from axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops import ParallelExperts
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
    parallel_linear,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
    ScatterMoELoRA,
    parallel_linear_lora,
)

__all__ = [
    "layers",
    "ParallelExperts",
    "flatten_sort_count",
    "parallel_linear",
    "ScatterMoELoRA",
    "parallel_linear_lora",
    "lora_ops",
]
from axolotl.integrations.kernels.libs.scattermoe_lora.multi_lora import (  # noqa: E402,F401
    ScatterMoEMultiLoRA,
    build_multilora_routing,
    scatter2scatter_multilora,
)
