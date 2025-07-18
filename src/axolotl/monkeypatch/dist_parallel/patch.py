"""Monkey patch for Accelerate to add support for ND parallelism."""

import accelerate
import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_prepare_device_mesh(
    sequence_parallel_degree: int = 1,
    dp_shard_size: int | None = None,
    fsdp: bool = False,
):
    """Patches the `Accelerator._prepare_device_mesh` method to create a device mesh
    that includes sequence parallelism with the specified degree.

    Args:
        sequence_parallel_degree: The degree of sequence parallelism to use.
        dp_shard_size: The number of data parallel replicas.
        fsdp: Whether to use FSDP.
    """

    def _prepare_device_mesh(self):
        """Prepare the device mesh for distributed training. The dataloader will
        determine how to load data based on the device mesh.
        """
        if self.state.torch_tp_plugin:
            return self.state.torch_tp_plugin.torch_device_mesh
        if (
            self.distributed_type == accelerate.accelerator.DistributedType.DEEPSPEED
            and hasattr(self.state, "ds_device_mesh")
        ):
            return self.state.ds_device_mesh

        # Create device mesh with sequence parallelism
        world_size = dist.get_world_size()

        dp_replicate_size = 1
        dp_shard_size_ = dp_shard_size  # workaround for closure modifying variable

        # if dp_shard_size isn't defined, we use assume there are no dp_replicas
        if dp_shard_size_ is None:
            dp_shard_size_ = world_size // sequence_parallel_degree
        else:
            dp_replicate_size = world_size // (
                dp_shard_size_ * sequence_parallel_degree
            )

        if dp_shard_size_ == 1:
            raise ValueError("dp_shard_size must be greater than 1")

        mesh_shape: tuple[int, ...] = ()
        mesh_dim_names: tuple[str, ...] = ()
        if dp_replicate_size > 1:
            mesh_shape += (dp_replicate_size,)
            mesh_dim_names += ("dp_replicate",)
        mesh_shape += (dp_shard_size,)
        mesh_dim_names += ("dp",) if not fsdp else ("fsdp",)
        if sequence_parallel_degree > 1:
            mesh_shape += (sequence_parallel_degree,)
            mesh_dim_names += ("sp",)
        device_ids = list(range(world_size))

        # NOTE: We use "cp" instead of "sp" to match the PyTorch native "context
        # parallelism" implementation naming.
        # NOTE: We have a simplified FSDP handling here; i.e., if FSDP is enabled, we
        # only use "fsdp" and "cp" for the device mesh.
        return dist.DeviceMesh(
            "cuda",
            torch.tensor(device_ids).reshape(mesh_shape),
            mesh_dim_names=mesh_dim_names,
        )

    # Replace the original method with our new method
    # pylint: disable=protected-access
    accelerate.accelerator.Accelerator._prepare_device_mesh = _prepare_device_mesh

    LOG.info(
        "Successfully patched Accelerator._prepare_device_mesh "
        f"with sequence_parallel_degree={sequence_parallel_degree} and "
        f"dp_shard_size={dp_shard_size} "
    )
