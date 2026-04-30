# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the Expert-Parallel (DeepEP) plugin."""

from typing import Literal

from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ExpertParallelArgs(BaseModel):
    """Input args for the expert_parallel plugin. See the integration README."""

    expert_parallel_size: int = 1
    """Number of EP ranks. 1 = disabled (default), > 1 = enabled."""

    expert_parallel_backend: Literal["deep_ep"] = "deep_ep"

    expert_parallel_num_nvl_bytes: int = 256 << 20

    expert_parallel_num_rdma_bytes: int = 0

    expert_parallel_fallback_on_unsupported: bool = True

    @model_validator(mode="after")
    def _validate(self):
        if self.expert_parallel_size < 1:
            raise ValueError(
                f"expert_parallel_size must be >= 1 (got {self.expert_parallel_size!r}). "
                f"Use 1 to disable EP."
            )

        if self.expert_parallel_size > 1 and self.expert_parallel_num_rdma_bytes != 0:
            LOG.warning(
                "expert_parallel_num_rdma_bytes != 0 — RDMA path requires "
                "Hopper + IBGDA-capable InfiniBand. Will fail on Ampere/intranode."
            )

        return self
