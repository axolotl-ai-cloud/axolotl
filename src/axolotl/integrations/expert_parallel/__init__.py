# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-Parallel integration for axolotl.

Replaces the dispatch/combine path in transformers MoE blocks with token-parallel
dispatch, on either of two backends (`expert_parallel_backend`): DeepEP's fused
kernels (`deep_ep`), or plain `all_to_all_single` over the EP process group (`torch`,
any NCCL/gloo fabric, no extra build). Registers eight names in
`transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`, one per backend x local kernel:

- `deep_ep` / `torch_ep_eager`             — eager local expert MLP (reference)
- `deep_ep_grouped_mm` / `torch_ep_grouped_mm` — transformers' grouped_mm kernel (default)
- `deep_ep_scattermoe` / `torch_ep_scattermoe` — axolotl's ScatterMoE kernel
- `deep_ep_sonicmoe` / `torch_ep_sonicmoe`     — axolotl's SonicMoE kernel

See the integration README for backend selection and requirements.
"""

from .args import ExpertParallelArgs
from .plugin import ExpertParallelPlugin

__all__ = [
    "ExpertParallelArgs",
    "ExpertParallelPlugin",
]
