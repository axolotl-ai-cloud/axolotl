"""
utility helpers for distributed checks
"""
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from accelerate import Accelerator

accelerate = None  # pylint: disable=invalid-name


def load_accelerate():
    global accelerate  # pylint: disable=global-statement
    accelerate = Accelerator()


def is_distributed():
    """
    Check if distributed training is initialized.
    """
    global accelerate  # pylint: disable=global-statement
    if not accelerate:
        accelerate = Accelerator()
    return dist.is_available() and dist.is_initialized()


def barrier():
    """
    Acts as a barrier to wait for all processes. This ensures that all processes
    reach the barrier before proceeding further.
    """
    if is_distributed():
        dist.barrier()


def is_main_process():
    """
    Check if the current process is the main process.
    If not in distributed mode, always return True.
    """
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def get_world_size():
    return int(os.getenv("WORLD_SIZE", "1"))


@contextmanager
def zero_first(is_main):
    """
    runs the wrapped context so that rank 0 runs first before other ranks
    """
    if not is_main:  # other ranks wait first
        barrier()
    yield
    if is_main:  # then rank 0 waits after it has run the context
        barrier()


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
    if not is_distributed():
        return [value_scalar]
    value_tensor = torch.tensor(value_scalar, device=dist.get_rank()).float()

    if not is_main_process():
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
