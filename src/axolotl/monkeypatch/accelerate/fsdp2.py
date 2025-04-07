"""
monkeypatch for accelerate fsdp2 fix when modifying ordereddict during interation
"""

import sys

import torch


def fsdp2_load_full_state_dict(accelerator, model: torch.nn.Module, full_sd: dict):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to load the state dict into
        full_sd (`dict`): The full state dict to load, can only be on rank 0
    """
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor

    sharded_sd = model.state_dict()
    if accelerator.is_main_process:
        # Create a list of items to iterate over before modifying the dictionary
        items_to_process = list(zip(full_sd.items(), sharded_sd.values()))
        for (param_name, full_param), sharded_param in items_to_process:
            full_param = full_param.detach().cuda()
            mesh = sharded_param.device_mesh
            dist.broadcast(full_param, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_param, mesh, sharded_param.placements
            )
            sharded_sd[param_name] = sharded_tensor
    else:
        # Create a list of items to iterate over before modifying the dictionary
        items_to_process = list(sharded_sd.items())
        for param_name, sharded_param in items_to_process:
            full_tensor = torch.empty(
                sharded_param.size(), device="cuda", dtype=sharded_param.dtype
            )
            mesh = sharded_param.device_mesh
            dist.broadcast(full_tensor, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_param.placements
            )
            sharded_sd[param_name] = sharded_tensor

    model.load_state_dict(sharded_sd)


def patch_accelerate_fsdp_utils():
    from accelerate.utils import fsdp_utils

    fsdp_utils.fsdp2_load_full_state_dict = fsdp2_load_full_state_dict
    setattr(
        sys.modules["accelerate.utils.fsdp_utils"],
        "fsdp2_load_full_state_dict",
        fsdp2_load_full_state_dict,
    )
