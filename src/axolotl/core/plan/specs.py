from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PLAN_VERSION = "1.0"


@dataclass(frozen=True, kw_only=True, slots=True)
class MeshSpec:
    """Parallel layout in the convention torchtitan and transformers' DistributedConfig share.

    The world is ``dp_replicate * dp_shard * cp * tp * pp``. ``dp_shard`` is the dense FSDP
    degree (transformers' ``fsdp_size``); expert parallelism regroups the same ranks and is not
    a factor of the world size. Routed experts are EP-sharded over ``ep`` and FSDP-split over
    ``efsdp``.
    """

    world_size: int
    world_size_source: Literal["env", "declared"]
    device_kind: str
    dist: Literal["single", "ddp", "fsdp2", "deepspeed"]
    dp_replicate: int
    dp_shard: int  # dense FSDP degree (D-16)
    cp: int
    cp_impl: Literal["none", "ring", "ulysses"]  # ulysses under DeepSpeed
    tp: int
    ep: int
    pp: int = 1  # always 1 in v1
    # dp_shard // ep (ep_within_dp_shard) or dp_shard * cp * tp // ep (ep_within_dp_shard_cp_tp)
    efsdp: int
    expert_placement: Literal["none", "ep_within_dp_shard", "ep_within_dp_shard_cp_tp"]
    legacy_spelling: bool
