"""
utility helpers for distributed checks
"""
from contextlib import contextmanager

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
