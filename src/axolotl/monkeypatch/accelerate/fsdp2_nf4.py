"""Load CPU-staged NF4 state without allocating full GPU replicas."""

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from axolotl.utils.nf4_loading import nf4_phase


@nf4_phase("NF4 parameter shard distribution")
def load_staged_nf4_state(
    accelerator,
    model: torch.nn.Module,
    full_state: dict[str, torch.Tensor],
    offload_to_cpu: bool = False,
) -> None:
    """Consume rank-zero CPU state by broadcasting bounded shards into a wrapped model."""
    state = _distribute_nf4_state(
        accelerator, model.state_dict(), full_state, offload_to_cpu
    )
    model.load_state_dict(state, assign=True, strict=True)


def _agree_distribution_plan(accelerator, targets, device, full_state):
    """Settle the transfer order and per-parameter layout on rank zero for every rank.

    Each rank deriving its own plan from its own ``state_dict`` is what makes a
    mismatch hang: the ranks issue different collectives and wait on each other
    forever. Distributing rank zero's plan, and exchanging every rank's verdict on
    it before the first transfer, turns any disagreement into a raise on all ranks.
    """
    from torch.distributed.tensor import DTensor

    def describe(target):
        return (
            isinstance(target, DTensor),
            tuple(target.shape),
            str(target.dtype),
        )

    payload = [
        [(name, *describe(target)) for name, target in targets.items()]
        if accelerator.is_main_process
        else None
    ]
    dist.broadcast_object_list(payload, src=0, device=device)
    plan = payload[0]

    error = None
    local = {name: describe(target) for name, target in targets.items()}
    for name, sharded, shape, dtype in plan:
        if name not in local:
            error = f"CPU-staged NF4 distribution: rank is missing {name}"
        elif local[name] != (sharded, shape, dtype):
            error = (
                f"CPU-staged NF4 distribution disagrees on {name}: "
                f"rank zero has {(sharded, shape, dtype)}, this rank has {local[name]}"
            )
        elif accelerator.is_main_process:
            source = full_state.get(name)
            if source is None:
                error = f"CPU-staged NF4 distribution: no source for {name}"
            elif tuple(source.shape) != shape:
                error = (
                    f"CPU-staged NF4 distribution: {name} is {tuple(source.shape)} on "
                    f"rank zero but {shape} in the wrapped model"
                )
        if error:
            break
    if error is None and len(local) != len(plan):
        extra = sorted(set(local) - {name for name, *_ in plan})
        error = f"CPU-staged NF4 distribution: rank has extra keys {extra}"

    statuses: list = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, error)
    failures = [f"rank {rank}: {msg}" for rank, msg in enumerate(statuses) if msg]
    if failures:
        raise ValueError("; ".join(failures))
    return plan


def _shard_bounds(rows: int, size: int) -> list[tuple[int, int]]:
    """Reproduce ``torch.chunk(dim=0)`` boundaries, empty tails included."""
    height = (rows + size - 1) // size
    starts = [min(owner * height, rows) for owner in range(size)]
    return [(start, min(start + height, rows)) for start in starts]


def _distribute_nf4_state(accelerator, targets, full_state, offload_to_cpu=None):
    """Distribute selected tensors, preserving target devices unless offload is specified."""
    from torch.distributed.tensor import DTensor, Shard

    state = {}
    device = accelerator.device
    world = dist.get_world_size()
    rank = dist.get_rank()
    plan = _agree_distribution_plan(accelerator, targets, device, full_state)
    for name, sharded, _, _ in tqdm(
        plan,
        desc="Distributing NF4 tensors",
        disable=not accelerator.is_main_process,
        mininterval=5,
    ):
        target = targets[name]
        source = None
        if accelerator.is_main_process:
            source = full_state[name].to(target.dtype)
        if sharded:
            mesh = target.device_mesh
            if mesh.ndim != 1 or target.placements != (Shard(0),):
                raise ValueError(
                    "CPU-staged NF4 currently requires a one-dimensional Shard(0) mesh"
                )
            if dist.get_process_group_ranks(mesh.get_group()) != list(range(world)):
                raise ValueError(
                    "CPU-staged NF4 requires the full data-parallel world mesh"
                )
            size = mesh.size()
            bounds = _shard_bounds(target.shape[0], size)
            # FSDP2 pads every shard up to the first chunk's height, so scattering
            # at that height is both the layout it wants and the equal-size chunks
            # scatter demands
            _, height = bounds[0]
            start, end = bounds[rank]
            chunks = None
            if accelerator.is_main_process:
                padded = torch.zeros(
                    (height * size, *target.shape[1:]),
                    dtype=target.dtype,
                    device=device,
                )
                padded[: target.shape[0]].copy_(source)
                chunks = [
                    padded.narrow(0, owner * height, height) for owner in range(size)
                ]
                del padded
            transfer = torch.empty(
                (height, *target.shape[1:]), dtype=target.dtype, device=device
            )
            # the default group for every transfer: the mesh group is the same
            # member set, but a second communicator is a second ordering to keep
            # in step and desynchronizes the ranks when a branch disagrees
            dist.scatter(transfer, chunks, src=0)
            del chunks
            local = transfer[: end - start]
            del transfer
            value = DTensor.from_local(
                local,
                mesh,
                target.placements,
                run_check=False,
                shape=target.shape,
                stride=target.stride(),
            )
            if offload_to_cpu is None:
                value = value.to(target.device)
            elif offload_to_cpu:
                value = value.cpu()
        else:
            value = (
                source.to(device)
                if accelerator.is_main_process
                else torch.empty_like(target, device=device)
            )
            dist.broadcast(value, src=0)
            if offload_to_cpu is None:
                value = value.to(target.device)
        state[name] = value
        full_state.pop(name, None)
    return state


def patch_nf4_adapter_state() -> None:
    """Preserve FSDP2 gather/broadcast semantics for full adapter checkpoints."""
    from dataclasses import replace
    from functools import wraps
    from types import SimpleNamespace

    from accelerate.utils import fsdp_utils
    from peft import get_peft_model_state_dict
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    original_get = fsdp_utils._get_model_state_dict
    original_set = fsdp_utils._set_model_state_dict
    if getattr(original_get, "_axolotl_nf4", False):
        return

    def needs_full_state(model, adapter_only, options):
        return (
            getattr(model, "_axolotl_staged_nf4", False)
            and adapter_only
            and options is not None
            and options.full_state_dict
        )

    @wraps(original_get)
    def get_state(model, adapter_only=False, sd_options=None):
        if not needs_full_state(model, adapter_only, sd_options):
            return original_get(model, adapter_only=adapter_only, sd_options=sd_options)
        state = get_model_state_dict(
            model, options=replace(sd_options, ignore_frozen_params=True)
        )
        return get_peft_model_state_dict(
            model, state_dict=state, adapter_name=model.active_adapter
        )

    @wraps(original_set)
    def set_state(model, state_dict, adapter_only=False, sd_options=None):
        if not needs_full_state(model, adapter_only, sd_options):
            return original_set(
                model, state_dict, adapter_only=adapter_only, sd_options=sd_options
            )
        adapter = model.active_adapter
        local_state = get_model_state_dict(
            model, options=StateDictOptions(ignore_frozen_params=True)
        )
        names = {name.replace(f".{adapter}.", "."): name for name in local_state}
        adapter_state = get_peft_model_state_dict(
            model, state_dict=local_state, adapter_name=adapter
        )
        targets = {names[name]: local_state[names[name]] for name in adapter_state}
        state = {names[name]: value for name, value in state_dict.items()}
        target = next(iter(targets.values()))
        accelerator = SimpleNamespace(
            device=torch.device(target.device_mesh.device_type),
            is_main_process=dist.get_rank() == 0,
        )
        state = _distribute_nf4_state(accelerator, targets, state)
        return set_model_state_dict(
            model,
            state,
            options=replace(
                sd_options,
                strict=False,
                full_state_dict=False,
                broadcast_from_rank0=False,
            ),
        )

    get_state._axolotl_nf4 = True
    fsdp_utils._get_model_state_dict = get_state
    fsdp_utils._set_model_state_dict = set_state
