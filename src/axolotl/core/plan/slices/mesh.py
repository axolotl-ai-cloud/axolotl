"""Parallel layout: config degrees -> ``MeshSpec``.

Two spellings of ``dp_shard_size`` are accepted. The current one is the dense FSDP degree
(transformers' ``fsdp_size``), with expert parallelism regrouping the same ranks. The legacy one
is the extra split inside an expert-parallel group, where ``expert_parallel_size`` is a factor of
the world size. With ``expert_parallel_size > 1`` and ``dp_replicate_size`` never auto-filled,
at most one of them matches the world size, so detection needs no extra key.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from axolotl.core.plan.errors import PlanError
from axolotl.core.plan.specs import MeshSpec

READS = frozenset(
    {
        "tensor_parallel_size",
        "context_parallel_size",
        "expert_parallel_size",
        "dp_shard_size",
        "dp_replicate_size",
        "fsdp",
        "fsdp_config",
        "deepspeed",
    }
)
READS_PARTIAL = frozenset({"fsdp", "fsdp_config", "deepspeed"})

LAYOUT_ERROR = "PLAN-001"


def _degree(config: Mapping[str, object], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanError(
            LAYOUT_ERROR, key=key, value=value, hint="must be a positive integer"
        )
    return value


def _ranks(world_size: int, *factors: int) -> int | None:
    """``world_size`` divided by the product of ``factors``, or None if it doesn't divide."""
    product = 1
    for factor in factors:
        product *= factor
    return world_size // product if world_size % product == 0 else None


def _no_fit(key: str, value: object, world_size: int, hint: str) -> PlanError:
    return PlanError(
        LAYOUT_ERROR, key=key, value=value, hint=f"on {world_size} ranks, {hint}"
    )


def _dense_rewrite(
    dense: int | None, ep: int, dp_shard: int, replicate_set: bool
) -> str:
    if dense is None or dense % ep:
        return (
            f"no dp_shard_size fits: expert_parallel_size ({ep}) must divide the dense FSDP "
            f"degree, world_size / (dp_replicate_size x context_parallel_size x "
            f"tensor_parallel_size)"
        )
    rewrite = f"write dp_shard_size: {dense} or leave it unset"
    if not replicate_set and dp_shard % ep == 0 and dense % dp_shard == 0:
        rewrite += (
            f", or keep dp_shard_size: {dp_shard} and set dp_replicate_size: "
            f"{dense // dp_shard}"
        )
    return rewrite


def _check_replica_composition(
    world_size: int, replicate: int, dense: int, tp: int, cp: int
) -> None:
    # accelerate's ParallelismConfig rejects TP/CP next to pure replication
    if replicate > 1 and dense == 1 and tp * cp > 1:
        raise _no_fit(
            "dp_replicate_size",
            replicate,
            world_size,
            "tensor/context parallelism cannot compose with pure data-parallel replicas; "
            "shard the remaining ranks with FSDP (dp_shard_size) instead",
        )


def _dist(
    config: Mapping[str, object], world_size: int
) -> Literal["single", "ddp", "fsdp2", "deepspeed"]:
    if config.get("deepspeed"):
        return "deepspeed"
    if config.get("fsdp") or config.get("fsdp_config"):
        return "fsdp2"
    return "ddp" if world_size > 1 else "single"


def resolve_mesh(
    config: Mapping[str, object],
    *,
    world_size: int,
    world_size_source: Literal["env", "declared"] = "env",
    device_kind: str = "cuda",
) -> MeshSpec:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise PlanError(LAYOUT_ERROR, key="world_size", value=world_size)

    tp = _degree(config, "tensor_parallel_size") or 1
    cp = _degree(config, "context_parallel_size") or 1
    ep = _degree(config, "expert_parallel_size") or 1
    dp_shard = _degree(config, "dp_shard_size")
    dp_replicate = _degree(config, "dp_replicate_size")
    dist = _dist(config, world_size)

    if ep > 1 and dist != "fsdp2":
        raise PlanError(
            LAYOUT_ERROR,
            key="expert_parallel_size",
            value=ep,
            allowed=("fsdp2",),
            hint="expert parallelism shards experts under FSDP2; set fsdp_version: 2",
        )
    if dp_shard is not None and dp_shard > 1 and dist != "fsdp2":
        raise PlanError(
            LAYOUT_ERROR,
            key="dp_shard_size",
            value=dp_shard,
            hint=(
                "DeepSpeed's ZeRO stage owns sharding; remove dp_shard_size"
                if dist == "deepspeed"
                else "dp_shard_size is the FSDP degree; set fsdp_version: 2 or remove it"
            ),
        )

    replicas = _ranks(world_size, tp, cp)
    if replicas is None:
        raise _no_fit(
            "tensor_parallel_size",
            tp,
            world_size,
            f"tensor_parallel_size x context_parallel_size ({tp * cp}) must divide the world size",
        )
    cp_impl: Literal["none", "ring", "ulysses"] = (
        "none" if cp == 1 else "ulysses" if dist == "deepspeed" else "ring"
    )

    if dist != "fsdp2":
        # ranks outside the TP/CP group replicate: DDP, or ZeRO's own sharding under DeepSpeed
        if dp_replicate is not None and dp_replicate > 1 and dp_replicate != replicas:
            raise _no_fit(
                "dp_replicate_size",
                dp_replicate,
                world_size,
                f"the ranks outside the tensor/context-parallel group replicate, so "
                f"dp_replicate_size must be {replicas} or unset",
            )
        # DeepSpeed's sequence parallelism is not accelerate's cp axis
        _check_replica_composition(
            world_size, replicas, 1, tp, 1 if cp_impl == "ulysses" else cp
        )
        return MeshSpec(
            world_size=world_size,
            world_size_source=world_size_source,
            device_kind=device_kind,
            dist=dist,
            dp_replicate=replicas,
            dp_shard=1,
            cp=cp,
            cp_impl=cp_impl,
            tp=tp,
            ep=1,
            efsdp=1,
            expert_placement="none",
            legacy_spelling=False,
        )

    replicate = dp_replicate or 1
    dense_unset = _ranks(world_size, replicate, tp, cp)
    legacy = False
    if dp_shard is None or (dp_shard == 1 and ep == 1):
        if dense_unset is None:
            raise _no_fit(
                "dp_replicate_size",
                replicate,
                world_size,
                f"dp_replicate_size x context_parallel_size x tensor_parallel_size "
                f"({replicate * tp * cp}) must divide the world size",
            )
        dense = dense_unset
    elif replicate * dp_shard * tp * cp == world_size:
        dense = dp_shard
    elif ep > 1 and replicate * dp_shard * tp * cp * ep == world_size:
        dense = dp_shard * ep
        legacy = True
    else:
        if ep > 1:
            hint = _dense_rewrite(dense_unset, ep, dp_shard, dp_replicate is not None)
        elif dense_unset is not None:
            hint = f"write dp_shard_size: {dense_unset} or leave it unset"
            fill = _ranks(world_size, dp_shard, tp, cp)
            if dp_replicate is None and fill is not None:
                hint += f", or keep it and set dp_replicate_size: {fill}"
        else:
            hint = (
                "dp_replicate_size x dp_shard_size x context_parallel_size x "
                "tensor_parallel_size must equal the world size"
            )
        raise _no_fit("dp_shard_size", dp_shard, world_size, hint)

    if dense % ep:
        raise _no_fit(
            "expert_parallel_size",
            ep,
            world_size,
            f"expert_parallel_size must divide the dense FSDP degree dp_shard ({dense}); "
            f"experts do not borrow context- or tensor-parallel ranks",
        )
    _check_replica_composition(world_size, replicate, dense, tp, cp)

    return MeshSpec(
        world_size=world_size,
        world_size_source=world_size_source,
        device_kind=device_kind,
        dist=dist,
        dp_replicate=replicate,
        dp_shard=dense,
        cp=cp,
        cp_impl=cp_impl,
        tp=tp,
        ep=ep,
        efsdp=dense // ep,
        expert_placement="ep_within_dp_shard" if ep > 1 else "none",
        legacy_spelling=legacy,
    )
