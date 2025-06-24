"""Ring attention group registration and flash attention patching.

Make use of the `ring-flash-attn` (https://github.com/zhuzilin/ring-flash-attention)
package, specifically the `hf_adapter.substitute_hf_flash_attn` function to patch in
their sequence parallel version of Flash Attention 2.

We also provide some patches for accelerate functions to prepare the dataloader for
sequence parallelism training.
"""

import inspect

import accelerate
import torch
import torch.distributed as dist

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RingAttnFunc

LOG = get_logger(__name__)


RING_ATTN_GROUP = None

ORIGINAL_PREPARE_DATALOADER_CODE = """            submesh_fsdp_size = 1
            submesh_dp_size = 1
            submesh_tp_size = 1
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            if "dp" in torch_device_mesh.mesh_dim_names:
                submesh_dp_size = torch_device_mesh["dp"].size()
            if "fsdp" in torch_device_mesh.mesh_dim_names:
                submesh_fsdp_size = torch_device_mesh["fsdp"].size()
            process_index = process_index // submesh_tp_size"""

NEW_PREPARE_DATALOADER_CODE = """            submesh_fsdp_size = 1
            submesh_dp_size = 1
            submesh_tp_size = 1
            submesh_cp_size = 1
            if "cp" in torch_device_mesh.mesh_dim_names:
                submesh_cp_size = torch_device_mesh["cp"].size()
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            if "dp" in torch_device_mesh.mesh_dim_names:
                submesh_dp_size = torch_device_mesh["dp"].size()
            if "fsdp" in torch_device_mesh.mesh_dim_names:
                submesh_fsdp_size = torch_device_mesh["fsdp"].size()
            process_index = process_index // (submesh_tp_size * submesh_cp_size)"""


def get_ring_attn_group() -> dist.ProcessGroup:
    """Getter for ring attention group on this rank."""
    if RING_ATTN_GROUP is None:
        raise RuntimeError("register_ring_attn() not yet called")
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: dist.ProcessGroup | None):
    """Setter for ring attention group on this rank."""
    global RING_ATTN_GROUP  # pylint: disable=global-statement
    RING_ATTN_GROUP = ring_attn_group


def register_ring_attn(
    sequence_parallel_degree: int,
    heads_k_stride: int | None,
    ring_attn_func: RingAttnFunc | None,
):
    """Create ring attention group and substitute flash attn with ring flash attn.

    Args:
        sequence_parallel_degree: Sequence parallelism factor.
        heads_k_stride: Sequence parallelism K head stride size. Passed through to
            `varlen_llama3` `ring_flash_attn` implementation.
        ring_attn_func: `ring_flash_attn` ring attention implemention. If sample
            packing is enabled, it must be a `varlen` function; otherwise, it must be a
            `batch` function.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        LOG.info(
            "Enabling ring attention sequence parallelism: "
            f"each sequence will be processed across {sequence_parallel_degree} GPUs"
        )

    assert sequence_parallel_degree <= world_size, (
        f"sequence_parallel_degree ({sequence_parallel_degree}) "
        f"must be less than or equal to world_size ({world_size})"
    )
    assert world_size % sequence_parallel_degree == 0, (
        f"sequence_parallel_degree ({sequence_parallel_degree}) "
        f"must evenly divide world_size ({world_size})"
    )

    # Assign ranks to sequence parallel groups
    group_assignments = {}
    for i in range(world_size // sequence_parallel_degree):
        ring_attn_ranks = list(
            range(
                i * sequence_parallel_degree,
                (i + 1) * sequence_parallel_degree,
            )
        )
        group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")

        # Track which GPUs are in which groups
        for r in ring_attn_ranks:
            group_assignments[r] = i

        if rank in ring_attn_ranks:
            set_ring_attn_group(group)

    # Log the GPU group assignments
    if rank == 0:
        LOG.info(f"Sequence parallel group assignments: {group_assignments}")

    if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
        from ring_flash_attn import substitute_hf_flash_attn

        substitute_hf_flash_attn(
            process_group=get_ring_attn_group(), heads_k_stride=heads_k_stride or 1
        )
    elif ring_attn_func is RingAttnFunc.BATCH_RING:
        from axolotl.monkeypatch.ring_attn.adapters.batch import (
            substitute_hf_flash_attn,
        )

        substitute_hf_flash_attn(
            process_group=get_ring_attn_group(),
            ring_attn_func=ring_attn_func,
        )


def update_ring_attn_params(position_ids: torch.Tensor | None):
    """
    Calculate the cumulative sequence lengths for the current forward pass and pass the
    value to the substituted `ring_flash_attn`.

    Args:
        position_ids: Optional tensor of position IDs (for sample packed data).
    """
    from ring_flash_attn import update_ring_flash_attn_params

    cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
    cu_seqlens = cu_seqlens.squeeze().to(device=torch.cuda.current_device())
    update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())


def patch_prepare_data_loader():
    """Patch `accelerate.data_loader.prepare_data_loader` to respect the SP degree.

    Raies:
        RuntimeError: If source code to patch does not exist.
    """
    original_fn = accelerate.data_loader.prepare_data_loader
    original_source = inspect.getsource(original_fn)

    if ORIGINAL_PREPARE_DATALOADER_CODE not in original_source:
        raise RuntimeError(
            "SP patch failed - target snippet not found. "
            "Check accelerate's version or update the patch."
        )

    patched_source = original_source.replace(
        ORIGINAL_PREPARE_DATALOADER_CODE, NEW_PREPARE_DATALOADER_CODE
    )

    # Create a new function from the patched source
    namespace = {}
    exec(  # pylint: disable=exec-used  # nosec B102
        patched_source, accelerate.data_loader.__dict__, namespace
    )
    patched_function = namespace["prepare_data_loader"]

    accelerate.data_loader.prepare_data_loader = patched_function
    LOG.info("Patched accelerate.data_loader.prepare_data_loader for SP support")


def patch_prepare_device_mesh(sequence_parallel_degree: int):
    """Patches the `Accelerator._prepare_device_mesh` method to create a device mesh
    that includes sequence parallelism with the specified degree.

    Args:
        sequence_parallel_degree (int): The degree of sequence parallelism to use.
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
        mesh_shape = (
            world_size // sequence_parallel_degree,
            sequence_parallel_degree,
        )
        device_ids = list(range(world_size))

        # Note that we use "cp" instead of "sp" to match the PyTorch native "context
        # parallelism" implementation naming
        return dist.DeviceMesh(
            "cuda",
            torch.tensor(device_ids).reshape(mesh_shape),
            mesh_dim_names=("dp", "cp"),
        )

    # Replace the original method with our new method
    # pylint: disable=protected-access
    accelerate.accelerator.Accelerator._prepare_device_mesh = _prepare_device_mesh

    LOG.info(
        "Successfully patched Accelerator._prepare_device_mesh "
        f"with sequence_parallel_degree={sequence_parallel_degree}"
    )
