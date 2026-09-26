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
any NCCL/gloo fabric, no extra build). Registers one name, `expert_parallel`, in
`transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`; it wraps whichever experts
implementation is configured (`use_scattermoe`, `use_sonicmoe`, or any registered
`experts_implementation`, default `grouped_mm`) around the dispatch and combine.

See the integration README for backend selection and requirements.
"""

from .args import ExpertParallelArgs
from .plugin import ExpertParallelPlugin

__all__ = [
    "ExpertParallelArgs",
    "ExpertParallelPlugin",
]
