"""
workaround to allow parallelism config for pure CP
"""

import os
import warnings

from accelerate import DistributedType


def _validate_accelerator(self, accelerator):
    _warnings = set()
    if not accelerator.multi_device and self.total_size == 1:
        # No distributed setup, valid parallelism config
        return

    # We need this to ensure DDP works
    if self.total_size == 1:
        self._set_size("dp_replicate", accelerator.num_processes)

    if self.total_size != accelerator.num_processes:
        raise ValueError(
            f"ParallelismConfig total_size ({self.total_size}) does not match "
            f"num_processes ({accelerator.num_processes}). Please adjust dp_replicate_size/ "
            f"dp_shard_size/tp_size/cp_size."
        )

    # allow parallelism config when not using fsdp if using pure context parallelism
    allow_parallelism_config = False

    if (
        self.cp_size > 1
        and self.dp_shard_size <= 1
        and os.environ.get("ACCELERATE_ALLOW_CP_STANDALONE", "false").lower() == "true"
    ):
        allow_parallelism_config = True

    if (
        self.total_size > 1
        and not allow_parallelism_config
        and not (accelerator.is_fsdp2 or accelerator.multi_device)
    ):
        raise ValueError(
            f"ParallelismConfig is only compatible DistributedType.FSDP (version 2) or DistributedType.Multi{{Device}}, but got {accelerator.distributed_type}."
        )

    for parallelism, size in self._sizes.items():
        if size == 1 and getattr(self, f"{parallelism}_handler", None) is not None:
            _warnings.add(
                f"ParallelismConfig.{parallelism}_handler is set, but {parallelism}_size is set to 1. This handler will be ignored."
            )

    if _warnings and accelerator.is_main_process:
        warnings.warn(
            "ParallelismConfig has the following warnings:\n" + "\n".join(_warnings),
            UserWarning,
            stacklevel=2,
        )


def patched_is_fsdp2(self) -> bool:
    """
    Patched version of is_fsdp2 that guards against a None fsdp_plugin.
    """
    # The new logic checks if fsdp_plugin exists before accessing its attributes
    return (
        self.distributed_type == DistributedType.FSDP
        and self.fsdp_plugin
        and self.fsdp_plugin.fsdp_version == 2
    )


def patch_parallelism_config():
    from accelerate.accelerator import AcceleratorState, ParallelismConfig

    ParallelismConfig._validate_accelerator = _validate_accelerator
    AcceleratorState.is_fsdp2 = property(patched_is_fsdp2)


def patch_prepare_cp():
    import functools

    import torch
    from accelerate import Accelerator

    def patched_prepare_cp(self, *args):
        if self.parallelism_config.cp_backend == "deepspeed":
            return args

        from accelerate.big_modeling import _attach_context_parallel_hooks
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method

        cp_comm_strategy = self.parallelism_config.cp_handler.cp_comm_strategy
        set_rotate_method(cp_comm_strategy)

        self._cp_context = functools.partial(
            context_parallel, mesh=self.torch_device_mesh["cp"]
        )

        for arg in args:
            if isinstance(arg, torch.nn.Module):
                _attach_context_parallel_hooks(arg)

        return args

    Accelerator._prepare_cp = patched_prepare_cp
