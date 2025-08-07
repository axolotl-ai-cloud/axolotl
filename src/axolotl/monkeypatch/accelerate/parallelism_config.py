"""
workaround to allow parallelism config for pure CP
"""

# pylint: disable=protected-access
import os
import warnings


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
        self.cp_size > 1  # pylint: disable=chained-comparison
        and self.dp_shard_size <= 1
        and os.environ.get("ACCELERATE_ALLOW_CP_STANDALONE", "false").lower() == "true"
    ):
        allow_parallelism_config = True

    if (
        self.total_size > 1
        and not (accelerator.is_fsdp2 or accelerator.multi_device)
        and not allow_parallelism_config
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
        )


def patch_parallelism_config():
    from accelerate.parallelism_config import ParallelismConfig

    ParallelismConfig._validate_accelerator = _validate_accelerator
