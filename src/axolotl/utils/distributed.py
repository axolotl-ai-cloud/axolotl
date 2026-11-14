"""Utilities for distributed functionality."""

import functools
import importlib.util
import os
import pickle  # nosec
from contextlib import contextmanager
from datetime import timedelta
from functools import lru_cache

import torch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import ParallelismConfig
from transformers.utils.import_utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
)

from axolotl.core.plan.slices.mesh import (
    READS as MESH_READS,
    READS_PARTIAL as MESH_READS_PARTIAL,
    resolve_mesh,
)
from axolotl.core.plan.specs import MeshSpec
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

distributed_state = None


@functools.cache
def is_tpu() -> bool:
    """Return True when TorchTPU (``torch_tpu``) is the active accelerator.

    Uses the compile-time accelerator check, which does not initialize the TPU
    runtime: a probe in the ``axolotl train`` parent process would hold the chip
    and starve the training process.
    """
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    accelerator = torch.accelerator.current_accelerator()
    return accelerator is not None and accelerator.type == "tpu"


def set_accelerate_tpu_device():
    # accelerate's default_device has no TPU branch and would fall back to CPU.
    os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "tpu:0")


def get_device_type() -> torch.device:
    if is_tpu():
        return torch.accelerator.current_accelerator()
    device = torch.device("cpu")
    if is_torch_cuda_available():
        device = torch.device("cuda")
    elif is_torch_mps_available():
        device = torch.device("mps")
    elif is_torch_npu_available():
        device = torch.device("npu")
    return device


def get_device_count() -> int:
    if is_tpu():
        return torch.accelerator.device_count()
    cur_device = get_device_type()
    if "cuda" in str(cur_device):
        return torch.cuda.device_count()
    if "npu" in str(cur_device):
        return torch.npu.device_count()
    return 1


def get_current_device() -> int:
    if is_tpu():
        return torch.accelerator.current_device_index()
    cur_device = get_device_type()
    if "cuda" in str(cur_device):
        return torch.cuda.current_device()
    if "npu" in str(cur_device):
        return torch.npu.current_device()
    return 0


def _backend_timeout(group, device) -> timedelta | None:
    try:
        return group._get_backend(device).options._timeout
    except Exception:  # noqa: BLE001  pylint: disable=broad-except
        return None


def _process_group_timeout() -> timedelta | None:
    """Best-effort read of the timeout the live process group was created with."""
    group = dist.group.WORLD
    for device in (get_device_type(), torch.device("cpu")):
        timeout = _backend_timeout(group, device)
        if timeout is not None:
            return timeout
    try:
        return dist.distributed_c10d._get_process_group_store(group).timeout
    except Exception:  # noqa: BLE001  pylint: disable=broad-except
        return None


def init_distributed_state():
    global distributed_state
    if distributed_state is not None:
        return
    if is_tpu():
        set_accelerate_tpu_device()
    from axolotl.utils.logging import get_logger

    log = get_logger(__name__)
    requested = int(os.environ.get("AXOLOTL_NCCL_TIMEOUT", 1800))
    # accelerate drops PartialState kwargs once a process group or PartialState exists
    preexisting = dist.is_initialized() or bool(PartialState._shared_state)
    try:
        distributed_state = PartialState(timeout=timedelta(seconds=requested))
    except ValueError as exc:
        log.warning("Could not initialize the distributed state: %s", exc)
        return
    if not preexisting:
        log.debug("Distributed timeout set to %ss", requested)
        return
    if not dist.is_initialized():
        return
    effective = _process_group_timeout()
    if effective is not None and effective >= timedelta(seconds=requested):
        return
    log.warning(
        "The requested distributed timeout of %ss was discarded: the process group was "
        "already initialized with a timeout of %s. Pass timeout= to "
        "torch.distributed.init_process_group before Axolotl initializes, or let Axolotl "
        "create the process group, for ddp_timeout / AXOLOTL_NCCL_TIMEOUT to take effect.",
        requested,
        effective if effective is not None else "unknown",
    )


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
    # the state outlives the process group: post-training hooks log after teardown
    if not dist.is_initialized():
        return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"
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


def _cfg_value(cfg, key):
    return cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)


def _written(cfg, key) -> bool:
    return (_cfg_value(cfg, key) or 1) > 1


def resolve_cfg_mesh(cfg, world_size: int | None = None) -> MeshSpec:
    """The parallel layout ``cfg`` builds on ``world_size`` ranks (default: ``WORLD_SIZE``).

    Raises ``PlanError`` (PLAN-001) for a layout that cannot fill the world size.
    """
    layout = tuple(
        sorted(
            (key, bool(value) if key in MESH_READS_PARTIAL else value)
            for key in MESH_READS
            for value in (_cfg_value(cfg, key),)
        )
    )
    return _resolve_layout(
        layout, get_world_size() if world_size is None else world_size
    )


@lru_cache(maxsize=32)
def _resolve_layout(layout: tuple, world_size: int) -> MeshSpec:
    config = dict(layout)
    if world_size == 1:
        # single-process commands (preprocess, inference, merge) run without the layout
        config = {key: config[key] for key in MESH_READS_PARTIAL}
    return resolve_mesh(config, world_size=world_size)


def _declared_layout(cfg) -> dict[str, int]:
    return {
        key: value
        for key in (
            "dp_replicate_size",
            "dp_shard_size",
            "context_parallel_size",
            "tensor_parallel_size",
            "expert_parallel_size",
        )
        if (value := _cfg_value(cfg, key) or 1) > 1
    }


def log_mesh_spelling(cfg, mesh: MeshSpec) -> None:
    if mesh.world_size == 1 and (declared := _declared_layout(cfg)):
        LOG.info("parallel layout %s is not applied in a single process", declared)
    if mesh.legacy_spelling:
        LOG.warning(
            "dp_shard_size: %d with expert_parallel_size: %d uses the deprecated spelling, "
            "where dp_shard_size is the split inside an expert-parallel group. dp_shard_size "
            "is now the dense FSDP degree (transformers' fsdp_size) and expert parallelism "
            "regroups the same ranks: write dp_shard_size: %d, or leave it unset, for this "
            "layout. The old spelling will be removed in a future release.",
            _cfg_value(cfg, "dp_shard_size"),
            mesh.ep,
            mesh.dp_shard,
            main_process_only=True,
        )


def _accelerate_dp_shard(mesh: MeshSpec) -> int:
    # accelerate's dp_shard axis is the split inside an EP group; its patched mesh adds `ep`
    return mesh.efsdp if mesh.ep > 1 else mesh.dp_shard


def _replicas_only(mesh: MeshSpec) -> bool:
    return mesh.dist in ("single", "ddp") or (
        mesh.dist == "deepspeed" and mesh.cp_impl != "ulysses"
    )


def mesh_to_parallelism_kwargs(
    mesh: MeshSpec, *, dp_replicate_written: bool = False
) -> dict[str, int | str]:
    """Lower a ``MeshSpec`` to accelerate ``ParallelismConfig`` kwargs.

    EP is not a kwarg: the patched ``ParallelismConfig`` reads ``PARALLELISM_CONFIG_EP_SIZE``.
    Without FSDP the replicas sit on a ``dp_shard`` axis unless ``dp_replicate_size`` was
    written, matching the meshes Axolotl has always built for DDP and DeepSpeed.
    """
    kwargs: dict[str, int | str] = {}
    if mesh.tp > 1:
        kwargs["tp_size"] = mesh.tp
    if mesh.cp_impl == "ulysses":
        kwargs["sp_size"] = mesh.cp
        kwargs["sp_backend"] = "deepspeed"
    elif mesh.cp > 1:
        kwargs["cp_size"] = mesh.cp
    if _replicas_only(mesh):
        if mesh.dp_replicate > 1:
            axis = "dp_replicate_size" if dp_replicate_written else "dp_shard_size"
            kwargs[axis] = mesh.dp_replicate
        return kwargs
    if mesh.dp_replicate > 1:
        kwargs["dp_replicate_size"] = mesh.dp_replicate
    if _accelerate_dp_shard(mesh) > 1:
        kwargs["dp_shard_size"] = _accelerate_dp_shard(mesh)
    return kwargs


def mesh_to_accelerate_env(
    mesh: MeshSpec,
    *,
    dp_shard_written: bool = False,
    dp_replicate_written: bool = False,
    sp_attn_implementation: str | None = None,
) -> dict[str, str]:
    """Lower a ``MeshSpec`` to the ``PARALLELISM_CONFIG_*`` environment accelerate builds its
    ``ParallelismConfig`` from, plus ``ACCELERATE_USE_PARALLELISM_CONFIG`` when it should.

    A layout that is only FSDP (or only replicas) gets no accelerate ``ParallelismConfig``
    unless ``dp_shard_size`` (or ``dp_replicate_size``) was written, as before.
    """
    env: dict[str, str] = {}
    if mesh.tp > 1:
        env["PARALLELISM_CONFIG_TP_SIZE"] = str(mesh.tp)
    if mesh.cp_impl == "ulysses":
        env["PARALLELISM_CONFIG_SP_SIZE"] = str(mesh.cp)
        env["PARALLELISM_CONFIG_SP_BACKEND"] = "deepspeed"
        env["PARALLELISM_CONFIG_SP_SEQ_LENGTH_IS_VARIABLE"] = "true"
        env["PARALLELISM_CONFIG_SP_ATTN_IMPLEMENTATION"] = (
            sp_attn_implementation or "sdpa"
        )
    elif mesh.cp > 1:
        env["PARALLELISM_CONFIG_CP_SIZE"] = str(mesh.cp)
        env["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    if mesh.ep > 1:
        env["PARALLELISM_CONFIG_EP_SIZE"] = str(mesh.ep)

    if mesh.dp_replicate > 1 and (
        not _replicas_only(mesh) or mesh.cp_impl == "ulysses" or dp_replicate_written
    ):
        env["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] = str(mesh.dp_replicate)
    shard = _accelerate_dp_shard(mesh)
    composed = mesh.tp > 1 or mesh.cp > 1 or mesh.ep > 1 or mesh.dp_replicate > 1
    if shard > 1 and (dp_shard_written or composed):
        env["PARALLELISM_CONFIG_DP_SHARD_SIZE"] = str(shard)

    if any(key != "PARALLELISM_CONFIG_EP_SIZE" for key in env):
        env["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
    return env


def build_parallelism_config(cfg):
    pc_kwargs = mesh_to_parallelism_kwargs(
        resolve_cfg_mesh(cfg),
        dp_replicate_written=_written(cfg, "dp_replicate_size"),
    )
    if pc_kwargs:
        parallelism_config = ParallelismConfig(**pc_kwargs)
        device_mesh = parallelism_config.build_device_mesh("cuda")
        return parallelism_config, device_mesh
    return None, None
