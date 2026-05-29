# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-Parallel (DeepEP) integration for axolotl.

Replaces the dispatch/combine path in transformers MoE blocks with DeepEP's
fused kernels. Registers four names in `transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`:

- `deep_ep`               — eager local expert MLP (reference)
- `deep_ep_grouped_mm`    — transformers' grouped_mm kernel (default)
- `deep_ep_scattermoe`    — axolotl's ScatterMoE kernel
- `deep_ep_sonicmoe`      — axolotl's SonicMoE kernel
"""

from .args import ExpertParallelArgs
from .plugin import ExpertParallelPlugin

__all__ = [
    "ExpertParallelArgs",
    "ExpertParallelPlugin",
]
