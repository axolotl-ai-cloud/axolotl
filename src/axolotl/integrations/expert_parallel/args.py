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
    """Input args for the expert_parallel plugin.

    EP is enabled by setting `expert_parallel_size > 1` — same UX as
    `tensor_parallel_size` and `dp_shard_size`. No separate enable flag.

    The plugin **auto-composes** with `experts_implementation` and
    `use_scattermoe` rather than introducing its own kernel-selection field.
    The user picks a kernel via standard axolotl config and the plugin
    transparently upgrades it to the DeepEP-wrapped variant when EP is on.

    Composition with FSDP (4+ GPUs) follows the same pattern as TP+FSDP:
    set `expert_parallel_size`, `dp_shard_size`, etc. such that their product
    equals world_size. The plugin builds a 2D mesh `(ep, dp_shard)` with the
    EP axis orthogonal to FSDP's contiguous dp_shard groups.

    Mapping for the local-experts kernel under EP:

    | User setting                                    | Effective kernel        |
    |-------------------------------------------------|-------------------------|
    | `use_scattermoe: true`                          | `deep_ep_scattermoe`    |
    | `use_sonicmoe: true`                            | `deep_ep_sonicmoe`      |
    | `experts_implementation: grouped_mm/batched_mm` | `deep_ep_grouped_mm`    |
    | `experts_implementation: eager`                 | `deep_ep` (eager local) |
    | (unset)                                         | `deep_ep_grouped_mm`    |
    """

    expert_parallel_size: int = 1
    """Number of EP ranks. 1 = disabled (default), > 1 = enabled.

    Constraints when > 1:
    - `world_size == expert_parallel_size * dp_shard_size * tensor_parallel_size * context_parallel_size`
    - For pure EP (no FSDP), the simplest valid setup is `expert_parallel_size == world_size`.
    - For EP × FSDP, set both `expert_parallel_size` and `dp_shard_size` such that
      their product matches world_size (e.g., world=4, ep=2, dp_shard=2).
    """

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
