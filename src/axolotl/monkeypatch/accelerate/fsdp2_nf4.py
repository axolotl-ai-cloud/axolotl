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
    from torch.distributed.tensor import DTensor, Shard

    state = {}
    device = accelerator.device
    for name, target in tqdm(
        model.state_dict().items(),
        desc="Distributing NF4 tensors",
        disable=not accelerator.is_main_process,
        mininterval=5,
    ):
        source = full_state.get(name) if accelerator.is_main_process else None
        if isinstance(target, DTensor):
            mesh = target.device_mesh
            if mesh.ndim != 1 or target.placements != (Shard(0),):
                raise ValueError(
                    "CPU-staged NF4 currently requires a one-dimensional Shard(0) mesh"
                )
            group = mesh.get_group()
            if dist.get_process_group_ranks(group) != list(
                range(dist.get_world_size())
            ):
                raise ValueError(
                    "CPU-staged NF4 requires the full data-parallel world mesh"
                )
            size = mesh.size()
            rank = dist.get_rank(group)
            rows = (target.shape[0] + size - 1) // size
            local = None
            for owner in range(size):
                start = min(owner * rows, target.shape[0])
                end = min(start + rows, target.shape[0])
                shape = (end - start, *target.shape[1:])
                if accelerator.is_main_process:
                    transfer = source[start:end].contiguous().to(device)
                else:
                    transfer = torch.empty(shape, dtype=target.dtype, device=device)
                dist.broadcast(transfer, src=0, group=group)
                if owner == rank:
                    local = transfer.cpu() if offload_to_cpu else transfer
                del transfer
            value = DTensor.from_local(
                local,
                mesh,
                target.placements,
                run_check=False,
                shape=target.shape,
                stride=target.stride(),
            )
        else:
            value = (
                source.to(device)
                if accelerator.is_main_process
                else torch.empty_like(target, device=device)
            )
            dist.broadcast(value, src=0)
        state[name] = value
        full_state.pop(name, None)
    model.load_state_dict(state, assign=True, strict=True)


def patch_nf4_adapter_state() -> None:
    """Preserve FSDP2 gather/broadcast semantics for full adapter checkpoints."""
    from dataclasses import replace
    from functools import wraps

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
        names = {
            name.replace(f".{adapter}.", "."): name
            for name in get_model_state_dict(
                model, options=StateDictOptions(ignore_frozen_params=True)
            )
        }
        state = {names[name]: value for name, value in state_dict.items()}
        return set_model_state_dict(
            model, state, options=replace(sd_options, strict=False)
        )

    get_state._axolotl_nf4 = True
    fsdp_utils._get_model_state_dict = get_state
    fsdp_utils._set_model_state_dict = set_state
