"""
monkeypatch for accelerate fsdp2 fix when modifying ordereddict during interation
"""

import logging
import sys

import torch

LOG = logging.getLogger(__name__)


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

    LOG.info("Broadcasting full state dict to all ranks...")
    sharded_sd = model.state_dict()
    param_names = sorted(sharded_sd.keys())
    for param_name in param_names:
        mesh = sharded_sd[param_name].device_mesh
        if accelerator.is_main_process:
            # Use the corresponding tensor from full_sd (assuming the key exists in full_sd)
            full_param = full_sd[param_name].detach().cuda()
            dist.broadcast(full_param, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_param, mesh, sharded_sd[param_name].placements
            )
            sharded_sd[param_name] = sharded_tensor
        else:
            # Prepare a tensor of matching shape and dtype
            full_tensor = torch.empty(
                sharded_sd[param_name].size(),
                device="cuda",
                dtype=sharded_sd[param_name].dtype,
            )
            dist.broadcast(full_tensor, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_sd[param_name].placements
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
