"""Ring attention group registration and flash attention patching.

Registers the context parallel process group from the device mesh, resolves the flash
attention kernels behind the model's `attn_implementation`, and swaps transformers'
flash attention entry point for the ring attention variants in this package. Also
provides the per-step varlen parameter update the sequence parallel context manager
calls before each forward pass.
"""

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from axolotl.monkeypatch.ring_attn.backends import resolve_flash_backend
from axolotl.monkeypatch.ring_attn.hf_adapter import (
    substitute_hf_flash_attn,
    update_ring_flash_attn_params,
)
from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RingAttnFunc

LOG = get_logger(__name__)

RING_ATTN_GROUP = None
RING_ATTN_FUNC: RingAttnFunc | None = None


def get_ring_attn_group() -> dist.ProcessGroup:
    """Getter for ring attention group on this rank."""
    if RING_ATTN_GROUP is None:
        raise RuntimeError("register_ring_attn_from_device_mesh() not yet called")
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: dist.ProcessGroup | None):
    """Setter for ring attention group on this rank."""
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = ring_attn_group


def register_ring_attn_from_device_mesh(
    device_mesh: "DeviceMesh",
    context_parallel_dim: tuple[str, ...],
    heads_k_stride: int | None,
    ring_attn_func: RingAttnFunc | None,
    attn_implementation: str | None = None,
):
    """Create the ring attention group from the DeviceMesh and patch in ring attention.

    Args:
        device_mesh: DeviceMesh object containing the parallelism topology.
        context_parallel_dim: Name of the sequence parallel dimension in the device mesh.
        heads_k_stride: Sequence parallelism K head stride size. Passed through to the
            `varlen_llama3` implementation.
        ring_attn_func: Ring attention implementation. If sample packing is enabled, it
            must be a `varlen` function; otherwise, it must be a `batch` function. None
            leaves attention untouched (e.g. GLM DSA kernels own context parallelism).
        attn_implementation: The model's resolved attention backend; selects the flash
            attention kernels (FA2/FA3/FA4, package or kernels hub) the ring
            functions run on.
    """
    global RING_ATTN_FUNC

    rank = dist.get_rank()

    LOG.info(
        f"Enabling ring attention sequence parallelism using DeviceMesh "
        f"dimension '{context_parallel_dim}'",
    )

    try:
        sequence_mesh = device_mesh[context_parallel_dim]
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Dimension '{context_parallel_dim}' not found in device_mesh. "
            f"Available dimensions: {device_mesh.mesh_dim_names}"
        ) from e

    sequence_pg = sequence_mesh.get_group()
    context_parallel_size = sequence_mesh.size()

    if rank == 0:
        LOG.info(
            f"Sequence parallel degree: {context_parallel_size}, "
            f"mesh shape: {sequence_mesh.mesh.shape}"
        )

    if sequence_pg != dist.GroupMember.WORLD:
        ranks_in_group = dist.get_process_group_ranks(sequence_pg)
        LOG.info(f"Current sequence parallel group ranks: {ranks_in_group}")

    set_ring_attn_group(sequence_pg)
    RING_ATTN_FUNC = ring_attn_func

    if ring_attn_func is None:
        return

    backend = resolve_flash_backend(attn_implementation)
    LOG.info(
        f"Ring attention `{ring_attn_func.value}` running on {backend.name} kernels "
        f"({backend.source})"
    )
    substitute_hf_flash_attn(
        backend=backend,
        process_group=get_ring_attn_group(),
        ring_attn_func=ring_attn_func,
        heads_k_stride=heads_k_stride or 1,
    )


def update_ring_attn_params(position_ids: torch.Tensor | None):
    """
    Calculate the cumulative sequence lengths for the current forward pass and pass the
    value to the substituted varlen ring attention.

    Args:
        position_ids: Optional tensor of position IDs (for sample packed data).
    """
    if RING_ATTN_FUNC is not RingAttnFunc.VARLEN_LLAMA3:
        # Batch ring needs no per-step slices; with no ring function registered the
        # attention kernels derive their own cu_seqlens from position_ids.
        return

    cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
    cu_seqlens = cu_seqlens.squeeze().to(device=torch.cuda.current_device())
    update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
