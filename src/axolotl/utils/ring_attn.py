import torch.distributed as dist
from ring_flash_attn import substitute_hf_flash_attn

RING_ATTN_GROUP = None


def get_ring_attn_group():
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = ring_attn_group


def register_ring_attn(sequence_parallel_size):
    """
    Create ring attention group and substitute flash attention with ring flash
    attention.
    """
    if sequence_parallel_size == 1:
        return

    world_size = dist.get_world_size()
    assert world_size % sequence_parallel_size == 0, (
        f"sequence_parallel_size ({sequence_parallel_size}) "
        f"must evenly divide world_size ({world_size})"
    )

    for i in range(world_size // sequence_parallel_size):
        ring_attn_ranks = list(
            range(
                i * sequence_parallel_size,
                (i + 1) * sequence_parallel_size,
            )
        )
        group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")
        if dist.get_rank() in ring_attn_ranks:
            set_ring_attn_group(group)

    substitute_hf_flash_attn(get_ring_attn_group(), sequence_parallel_size)
