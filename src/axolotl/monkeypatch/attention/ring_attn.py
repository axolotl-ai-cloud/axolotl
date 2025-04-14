"""
Ring attention group registration and flash attention patching.

Make use of the `ring-flash-attn` (https://github.com/zhuzilin/ring-flash-attention)
package, specifically the `hf_adapter.substitute_hf_flash_attn` function to patch in
their sequence parallel version of Flash Attention 2.
"""

import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from axolotl.logging_config import configure_logging
from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids

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


def update_ring_attn_params(batch: dict[str, torch.Tensor]):
    """
    Calculate the cumulative sequence lengths for the current forward pass and pass the
    value to the substituted `ring_flash_attn`.

    Args:
        batch: A dictionary with a batch of data. May or may not contain `position_ids`
            data; if not, we compute it.
    """
    from ring_flash_attn import update_ring_flash_attn_params

    input_ids = batch["input_ids"]
    position_ids = batch.get("position_ids")
    if position_ids is None:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

    cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
    cu_seqlens = cu_seqlens.squeeze().to(device=torch.cuda.current_device())
    update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
