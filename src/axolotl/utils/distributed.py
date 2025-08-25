"""Utilities for distributed functionality."""

import os
import pickle  # nosec
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import ParallelismConfig
from transformers.utils.import_utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
)

distributed_state = None


def get_device_type() -> torch.device:
    device = torch.device("cpu")
    if is_torch_cuda_available():
        device = torch.device("cuda")
    elif is_torch_mps_available():
        device = torch.device("mps")
    elif is_torch_npu_available():
        device = torch.device("npu")
    return device


def get_device_count() -> int:
    cur_device = get_device_type()
    if "cuda" in str(cur_device):
        return torch.cuda.device_count()
    if "npu" in str(cur_device):
        return torch.npu.device_count()
    return 1


def get_current_device() -> int:
    cur_device = get_device_type()
    if "cuda" in str(cur_device):
        return torch.cuda.current_device()
    if "npu" in str(cur_device):
        return torch.npu.current_device()
    return 0


def init_distributed_state():
    global distributed_state
    if distributed_state is None:
        timeout = int(os.environ.get("AXOLOTL_NCCL_TIMEOUT", 1800))
        try:
            distributed_state = PartialState(timeout=timedelta(seconds=timeout))
        except ValueError:
            pass


def get_distributed_state() -> PartialState | None:
    return distributed_state


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    init_distributed_state()

    if distributed_state is None:
        return False

    return distributed_state.use_distributed and distributed_state.initialized


def barrier():
    """
    Acts as a barrier to wait for all processes. This ensures that all processes
    reach the barrier before proceeding further.
    """
    if is_distributed():
        dist.barrier()


def is_main_process() -> bool:
    """
    Check if the current process is the main process. If not in distributed mode,
    always return `True`.

    We use a simpler logic when the distributed state is not initialized: we just log
    on the 0-th local rank.

    Returns:
        `True` if the current process is the main process, `False` otherwise.
    """
    if get_distributed_state() is None:
        return os.environ.get("LOCAL_RANK", "0") == "0"
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def is_local_main_process() -> bool:
    if get_distributed_state() is None:
        return os.environ.get("LOCAL_RANK", "0") == "0"
    return PartialState().is_local_main_process


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


def cleanup_distributed():
    """
    Destroy process group if torch distributed is initialized. Called in training early
    termination or when training successfully completes.
    """
    # Ensure that all operations are completed before destroying the process group
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.xpu.is_available():
        torch.xpu.synchronize()

    # Destroy the process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def zero_first(is_main: bool):
    """
    runs the wrapped context so that rank 0 runs first before other ranks
    """
    if not is_main:  # other ranks wait first
        barrier()
    yield
    if is_main:  # then rank 0 waits after it has run the context
        barrier()


def gather_scalar_from_all_ranks(fn, world_size=1):
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
    value_tensor = torch.tensor(
        value_scalar, device=f"{get_device_type()}:{get_current_device()}"
    ).float()

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


def broadcast_dict(vals: dict):
    if not is_distributed():
        return vals

    cur_device = get_device_type()
    if is_main_process():
        data_byte = pickle.dumps(vals)
        data_tensor = torch.ByteTensor(list(data_byte)).to(cur_device)
        data_size = torch.IntTensor([len(data_byte)]).to(cur_device)
    else:
        data_tensor = torch.empty([1024], dtype=torch.uint8, device=cur_device)
        data_size = torch.IntTensor([0]).to(cur_device)

    dist.broadcast(data_size, 0)
    if not is_main_process():
        # resize
        data_tensor = data_tensor.new_empty([data_size.item()])

    dist.broadcast(data_tensor, 0)

    if not is_main_process():
        data_list = data_tensor.cpu().tolist()
        data_byte = bytes(data_list[: data_size.item()])
        vals = pickle.loads(data_byte)  # nosec

    return vals


def compute_and_broadcast(fn):
    """
    Compute a value using the function 'fn' only on the specified rank (default is 0).
    The value is then broadcasted to all other ranks.

    Args:
    - fn (callable): A function that computes the value. This should not have any side effects.
    - rank (int, optional): The rank that computes the value. Default is 0.

    Returns:
    - The computed value (int or float).
    """
    cur_device = f"{get_device_type()}:{get_current_device()}"
    if is_main_process():
        value_scalar = fn()
        value_tensor = torch.tensor(
            value_scalar, device=cur_device, dtype=torch.float32
        )
    else:
        value_tensor = torch.tensor(
            0.0, device=cur_device, dtype=torch.float32
        )  # Placeholder tensor

    # Broadcast the tensor to all processes.
    barrier()
    dist.broadcast(value_tensor, src=0)

    # Convert the tensor back to its original type (int or float)
    if value_tensor == value_tensor.int():
        return int(value_tensor.item())
    return float(value_tensor.item())


def gather_from_all_ranks(fn, world_size=1):
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
    value_tensor = torch.tensor(
        value_scalar, device=f"{get_device_type()}:{get_current_device()}"
    ).float()

    # Placeholder tensor for gathering results
    if is_main_process():
        gathered_tensors = [torch.zeros_like(value_tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None

    dist.gather(value_tensor, gather_list=gathered_tensors, dst=0)

    if is_main_process():
        # Convert tensors back to their original type (int or float)
        gathered_values = []
        for tensor in gathered_tensors:
            if tensor == tensor.int():
                gathered_values.append(int(tensor.item()))
            else:
                gathered_values.append(float(tensor.item()))
        return gathered_values
    return None


def reduce_and_broadcast(fn1, fn2):
    """
    Run a callable 'fn1' on all ranks, gather the results, reduce them using 'fn2',
    and then broadcast the reduced result to all ranks.

    Args:
    - fn1 (callable): A function that computes the value on each rank.
    - fn2 (callable): A reduction function that takes a list of values and returns a single value.
    - world_size (int, optional): Total number of processes in the current distributed setup.

    Returns:
    - The reduced and broadcasted value.
    """

    # Gather values from all ranks using fn1
    if not is_distributed():
        return fn2([fn1()])

    gathered_values = gather_from_all_ranks(fn1, world_size=dist.get_world_size())

    # Use compute_and_broadcast to compute the reduced value on the main process
    # and then broadcast it to all ranks
    return compute_and_broadcast(lambda: fn2(gathered_values))


def build_parallelism_config(cfg):
    pc_kwargs = _get_parallel_config_kwargs(
        get_world_size(),
        cfg.tensor_parallel_size,
        cfg.context_parallel_size,
        cfg.dp_shard_size,
        cfg.dp_replicate_size,
        bool(cfg.fsdp or cfg.fsdp_config),
    )

    if pc_kwargs:
        parallelism_config = ParallelismConfig(
            **pc_kwargs,
        )
        device_mesh = parallelism_config.build_device_mesh("cuda")

        return parallelism_config, device_mesh
    return None, None


def _get_parallel_config_kwargs(
    world_size: int,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
    dp_shard_size: int | None = None,
    dp_replicate_size: int | None = None,
    is_fsdp: bool = False,
):
    pc_kwargs = {}
    remaining_world_size = world_size

    if tensor_parallel_size and tensor_parallel_size > 1:
        pc_kwargs["tp_size"] = tensor_parallel_size
        remaining_world_size = remaining_world_size // tensor_parallel_size

    if context_parallel_size and context_parallel_size > 1:
        pc_kwargs["cp_size"] = context_parallel_size
        remaining_world_size = remaining_world_size // context_parallel_size

    if dp_shard_size is None and dp_replicate_size in (None, 1):
        if remaining_world_size > 1:
            pc_kwargs["dp_shard_size"] = remaining_world_size
            remaining_world_size = 1

    if dp_replicate_size and dp_replicate_size > 1:
        pc_kwargs["dp_replicate_size"] = dp_replicate_size
        remaining_world_size = remaining_world_size // dp_replicate_size

    if remaining_world_size > 1 and dp_shard_size and dp_shard_size > 1:
        if not is_fsdp:
            raise ValueError(
                "dp_shard_size was configured without a corresponding fsdp_config! "
                "Please ensure you have configured FSDP using fsdp_config."
            )
        pc_kwargs["dp_shard_size"] = dp_shard_size
        remaining_world_size = remaining_world_size // dp_shard_size
        if remaining_world_size > 1 and "dp_replicate_size" not in pc_kwargs:
            pc_kwargs["dp_replicate_size"] = remaining_world_size
            remaining_world_size = 1

    if remaining_world_size > 1:
        if "dp_shard_size" not in pc_kwargs and is_fsdp:
            pc_kwargs["dp_shard_size"] = remaining_world_size
            remaining_world_size = 1

    if remaining_world_size > 1:
        raise ValueError(
            f"The configured parallelisms are incompatible with the current world size ({get_world_size()})!\n"
            f"{pc_kwargs}"
        )

    return pc_kwargs
