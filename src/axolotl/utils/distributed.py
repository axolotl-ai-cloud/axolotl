"""
utility helpers for distributed checks
"""
import torch.distributed as dist


def is_distributed():
    """
    Check if distributed training is initialized.
    """
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
