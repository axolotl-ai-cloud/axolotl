"""
Ring attention group registration and flash attention patching.

Make use of the `ring-flash-attn` (https://github.com/zhuzilin/ring-flash-attention)
package, specifically the `hf_adapter.substitute_hf_flash_attn` function to patch in
their sequence parallel version of Flash Attention 2.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.logging import get_logger

from axolotl.logging_config import configure_logging

configure_logging()
LOG = get_logger(__name__)

RING_ATTN_GROUP = None


def get_ring_attn_group() -> dist.ProcessGroup:
    """
    Getter for ring attention group on this rank.

    Returns:
        The process group for ring attention for this rank.
    """
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: dist.ProcessGroup | None):
    """
    Setter for ring attention group on this rank.

    Args:
        Process group for ring attention.
    """
    global RING_ATTN_GROUP  # pylint: disable=global-statement
    RING_ATTN_GROUP = ring_attn_group


def register_ring_attn(sequence_parallel_degree: int, heads_k_stride: int | None):
    """
    Create ring attention group and substitute flash attn with ring flash attn.

    Args:
        sequence_parallel_degree: Sequence parallelism factor.
        heads_k_stride: Sequence parallelism K head stride size. Passed
            through to `ring_flash_attn.substitute_hf_flash_attn`.
    """
    if get_ring_attn_group() is not None:
        LOG.info("Ring attention already registered, exiting early...")
        return

    LOG.info(
        "Enabling ring attention sequence parallelism: "
        f"each sequence will be processed across {sequence_parallel_degree} GPUs"
    )

    world_size = dist.get_world_size()
    assert sequence_parallel_degree <= world_size, (
        f"sequence_parallel_degree ({sequence_parallel_degree}) "
        f"must be less than or equal to world_size ({world_size})"
    )
    assert world_size % sequence_parallel_degree == 0, (
        f"sequence_parallel_degree ({sequence_parallel_degree}) "
        f"must evenly divide world_size ({world_size})"
    )

    # Detailed logging of group formation
    rank = dist.get_rank()
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

    if heads_k_stride is None:
        heads_k_stride = 1

    from ring_flash_attn import substitute_hf_flash_attn

    substitute_hf_flash_attn(
        process_group=get_ring_attn_group(), heads_k_stride=heads_k_stride
    )


def calculate_packed_seq_lens(position_ids: torch.Tensor) -> torch.Tensor:
    """
    Calculates lengths of packed sequences from position IDs tensor.

    Args:
        position_ids: A tensor of shape `[1, seq_len]` containing position IDs, where
            zeros indicate potential sequence starts.

    Returns:
        A tensor containing the lengths of each sequence in the packed format.
    """
    # Batch size must be 1 (checked in Pydantic config model validation)
    position_ids = position_ids.flatten()

    # Find where the position resets
    sequence_starts = torch.cat(
        [position_ids.new_ones(1), (position_ids[1:] == 0).to(torch.int)]
    )

    # Get all indices where sequence_starts
    potential_indices = torch.nonzero(sequence_starts).flatten()

    # Filter out indices where the next index also has a zero
    valid_indices = []
    for i, current_pos in enumerate(potential_indices):
        # Check if this is the last index or if the next element is not a zero
        if i == len(potential_indices) - 1:
            break
        valid_indices.append(current_pos)

    start_indices = torch.tensor(valid_indices, device=potential_indices.device)

    # Calculate packed sequence lengths
    if len(start_indices) > 1:
        packed_seq_lens = torch.diff(
            start_indices, append=torch.tensor([len(position_ids)])
        )
    else:
        packed_seq_lens = torch.tensor([len(position_ids)])

    return packed_seq_lens


def update_ring_attn_params(packed_seq_lens: torch.Tensor, total_seq_len: int):
    """
    Calculate the cumulative sequence lengths for the current forward pass and pass the
    value to the substituted ring_flash_attn.

    Logic borrowed from
    https://github.com/zhuzilin/OpenRLHF/blob/47f7cd8fc76de6d057d053251c1b55c00421cc24/openrlhf/models/ring_attn_utils.py#L43.

    Args:
        packed_seq_lens: Lengths of multipacked sequences.
        total_seq_len: Length of the full sequence.
    """
    cu_seqlens = torch.cumsum(
        packed_seq_lens.clone()
        .detach()
        .to(device=torch.cuda.current_device(), dtype=torch.int32),
        dim=-1,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len)

    from ring_flash_attn import update_ring_flash_attn_params

    update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
