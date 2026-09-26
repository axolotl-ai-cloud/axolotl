# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the Expert-Parallel plugin."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ExpertParallelArgs(BaseModel):
    """Input args for the expert_parallel plugin. See the integration README."""

    expert_parallel_size: int = 1
    """Number of EP ranks. 1 = disabled (default), > 1 = enabled."""

    expert_parallel_backend: Literal["auto", "deep_ep", "torch"] = "auto"
    """Token dispatch backend. ``deep_ep``: DeepEP fused kernels (NVLink/RDMA). ``torch``: plain
    ``all_to_all_single`` over the EP process group (any NCCL/gloo setup). ``auto``: ``deep_ep``
    when importable, else ``torch``."""

    expert_parallel_num_nvl_bytes: int = 256 << 20

    expert_parallel_num_rdma_bytes: int = 0

    expert_parallel_token_capacity: int | None = None
    """Max tokens routed to any single expert per forward; the lowest-weight excess (token,expert)
    assignments are dropped. DeepEP's intranode combine deadlocks once one expert is overloaded, and
    GLM-style routers concentrate more with depth — set this (e.g. 1024) for them. ``None`` = no cap."""

    expert_parallel_fallback_on_unsupported: bool = True

    expert_parallel_save_dispatch: bool = True
    """``torch`` backend under activation checkpointing (``gradient_checkpointing`` or FSDP2
    ``fsdp_config.activation_checkpointing``): save the
    dispatch/combine all-to-all outputs so recompute does not re-issue the collectives (costs the
    received rows per layer). The routing ``topk`` and host-side split counts are always saved."""

    expert_parallel_dispatch_chunks: int = Field(default=1, ge=1)
    """``torch`` backend only: split each MoE forward's tokens into this many chunks and pipeline
    them, so chunk ``i+1``'s dispatch all-to-all overlaps chunk ``i``'s expert GEMMs. ``1`` keeps
    the unchunked path. Overlaps the forward only. Each chunk adds collective and launch overhead,
    so this can be slower than ``1``; benchmark before raising it past ``2``."""

    @model_validator(mode="after")
    def _validate(self):
        if self.expert_parallel_size < 1:
            raise ValueError(
                f"expert_parallel_size must be >= 1 (got {self.expert_parallel_size!r}). "
                f"Use 1 to disable EP."
            )

        if (
            self.expert_parallel_token_capacity is not None
            and self.expert_parallel_token_capacity < 1
        ):
            raise ValueError(
                f"expert_parallel_token_capacity must be >= 1 when set "
                f"(got {self.expert_parallel_token_capacity!r}). Use None to disable the cap."
            )

        if self.expert_parallel_size > 1 and self.expert_parallel_num_rdma_bytes != 0:
            LOG.warning(
                "expert_parallel_num_rdma_bytes != 0 — RDMA path requires "
                "Hopper + IBGDA-capable InfiniBand. Will fail on Ampere/intranode."
            )

        return self
