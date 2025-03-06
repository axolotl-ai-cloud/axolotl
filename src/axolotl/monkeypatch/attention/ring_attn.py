"""Ring attention group registration and flash attention patching."""

from typing import Any

import torch.distributed as dist
from accelerate.logging import get_logger
from ring_flash_attn import substitute_hf_flash_attn

from axolotl.logging_config import configure_logging

configure_logging()
LOG = get_logger(__name__)

RING_ATTN_GROUP = None


def get_ring_attn_group() -> Any:
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: Any):
    global RING_ATTN_GROUP  # pylint: disable=global-statement
    RING_ATTN_GROUP = ring_attn_group


def register_ring_attn(sequence_parallel_size: int):
    """
    Create ring attention group and substitute flash attn with ring flash attn.

    Args:
        sequence_parallel_size: Sequence parallelism factor.
    """
    LOG.info(
        "Enabling ring attention sequence parallelism: "
        f"each sequence will be processed across {sequence_parallel_size} GPUs"
    )

    world_size = dist.get_world_size()
    assert world_size % sequence_parallel_size == 0, (
        f"sequence_parallel_size ({sequence_parallel_size}) "
        f"must evenly divide world_size ({world_size})"
    )

    # Detailed logging of group formation
    rank = dist.get_rank()
    group_assignments = {}

    for i in range(world_size // sequence_parallel_size):
        ring_attn_ranks = list(
            range(
                i * sequence_parallel_size,
                (i + 1) * sequence_parallel_size,
            )
        )
        group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")

        # Track which GPUs are in which groups
        for r in ring_attn_ranks:
            group_assignments[r] = i

        if rank in ring_attn_ranks:
            set_ring_attn_group(group)
            LOG.info(
                f"GPU {rank} assigned to sequence parallel group {i} with ranks {ring_attn_ranks}"
            )

    # Log the full group assignment structure
    if rank == 0:
        LOG.info(f"Sequence parallel group assignments: {group_assignments}")

    substitute_hf_flash_attn(get_ring_attn_group(), sequence_parallel_size)
