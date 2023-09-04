"""
utility helpers for distributed checks
"""
import torch
import torch.distributed as dist
from accelerate import DistributedType
from accelerate.state import PartialState
from accelerate.utils import wait_for_everyone

accelerate = None  # pylint: disable=invalid-name

state = PartialState()


def is_distributed():
    """
    Check if distributed training is initialized.
    """
    return state.distributed_type in (
        DistributedType.MULTI_GPU,
        DistributedType.MULTI_CPU,
        DistributedType.DEEPSPEED,
        DistributedType.FSDP,
    )


def barrier():
    """
    Acts as a barrier to wait for all processes. This ensures that all processes
    reach the barrier before proceeding further.
    """
    wait_for_everyone()


def is_main_process() -> bool:
    """
    Check if the current process is the main process.
    If not in distributed mode, always return True.
    """
    return state.is_main_process


def get_world_size() -> int:
    return state.num_processes


def gather_scalar_from_all_ranks(fn, world_size=1):  # pylint: disable=invalid-name
    """
    Run a callable 'fn' on all ranks and gather the results on the specified rank.

    Args:
    - fn (callable): A function that computes the value. This should not have any side effects.
    - rank (int, optional): The rank that gathers the values. Default is 0.
    - world_size (int, optional): Total number of processes in the current distributed setup.

    Returns:
    - A list of computed values from all ranks if on the gathering rank, otherwise None.
    """
    value_scalar = fn()
    value_tensor = torch.tensor(value_scalar, device=dist.get_rank()).float()

    if not state.is_main_process:
        dist.gather(value_tensor, dst=0)
    else:
        gathered_tensors = [torch.zeros_like(value_tensor) for _ in range(world_size)]
        dist.gather(value_tensor, gather_list=gathered_tensors, dst=0)

        # Convert tensors back to their original type (int or float)
        gathered_values = []
        for tensor in gathered_tensors:
            if tensor == tensor.int():
                gathered_values.append(int(tensor.item()))
            else:
                gathered_values.append(float(tensor.item()))
        return gathered_values
    return None
