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

    expert_parallel_save_dispatch: bool = False
    """``torch`` backend under activation checkpointing (``gradient_checkpointing`` or FSDP2
    ``fsdp_config.activation_checkpointing``): save the dispatch/combine all-to-all outputs so
    backward does not re-issue the collectives, at the cost of holding the received rows per
    layer (24 GiB on Qwen3-30B-A3B at 32k packed on 2 GPUs). Recomputing them cost 1.6 s/step on
    a PCIe-only pair and was a tie on an NVLink pair, so the default recomputes; enable this on
    bandwidth-starved links with memory to spare. The routing ``topk`` and host-side split
    counts are always saved."""

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

    @model_validator(mode="after")
    def _validate_topology(self):
        # only the merged input config carries the fsdp/mesh fields; standalone args skip
        if (
            self.expert_parallel_size <= 1
            or "fsdp_config" not in type(self).model_fields
        ):
            return self
        validate_expert_parallel_topology(self)
        return self


def validate_expert_parallel_topology(cfg) -> None:
    """Reject EP layouts the sharding, clipping and save paths do not handle."""
    ep_size = getattr(cfg, "expert_parallel_size", 1) or 1
    if ep_size <= 1:
        return

    fsdp_config = getattr(cfg, "fsdp_config", None)
    fsdp_version = getattr(cfg, "fsdp_version", None) or (
        getattr(fsdp_config, "fsdp_version", None) if fsdp_config else None
    )
    if not fsdp_config or fsdp_version != 2:
        raise ValueError(
            f"expert_parallel_size ({ep_size}) > 1 requires FSDP2: set fsdp_version: 2 "
            "and an fsdp_config block. Under DDP the expert LoRA sync, gradient clipping, "
            "gradient-accumulation scaling and checkpoint save are all wrong for sharded experts."
        )

    for key in ("state_dict_type", "final_state_dict_type"):
        value = getattr(fsdp_config, key, None)
        if key == "state_dict_type" and value is None:
            raise ValueError(
                f"expert_parallel_size ({ep_size}) > 1 requires fsdp_config.state_dict_type: "
                "FULL_STATE_DICT. accelerate defaults FSDP2 to SHARDED_STATE_DICT, whose "
                "checkpoints keep only EP group 0's experts (every group's expert shard has the "
                "same name, shape and offset, so DCP deduplicates them)."
            )
        if value is not None and value != "FULL_STATE_DICT":
            raise ValueError(
                f"expert_parallel_size ({ep_size}) > 1 requires fsdp_config.{key}: "
                f"FULL_STATE_DICT, got {value!r}. {value} checkpoints keep only EP group 0's "
                "experts; the full state dict gathers every EP group's experts before rank 0 writes."
            )
