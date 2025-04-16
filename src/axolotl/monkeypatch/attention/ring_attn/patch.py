"""
Ring attention group registration and flash attention patching.

Make use of the `ring-flash-attn` (https://github.com/zhuzilin/ring-flash-attention)
package, specifically the `hf_adapter.substitute_hf_flash_attn` function to patch in
their sequence parallel version of Flash Attention 2.
"""

from enum import Enum

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


class RingAttnFunc(str, Enum):
    """Enum class for supported `ring-flash-attn` implementations"""

    # VARLEN_RING = "varlen_ring"
    # VARLEN_ZIGZAG = "varlen_zigzag"
    VARLEN_LLAMA3 = "varlen_llama3"
    BATCH_RING = "batch_ring"
    BATCH_ZIGZAG = "batch_zigzag"
    BATCH_STRIPE = "batch_stripe"


def register_ring_attn(
    sequence_parallel_degree: int,
    heads_k_stride: int | None,
    ring_attn_func: RingAttnFunc | None,
):
    """
    Create ring attention group and substitute flash attn with ring flash attn.

    Args:
        sequence_parallel_degree: Sequence parallelism factor.
        heads_k_stride: Sequence parallelism K head stride size. Passed
            through to `ring_flash_attn.substitute_hf_flash_attn`.
        ring_attn_func: `ring_flash_attn` ring attention implemention. If sample
            packing is enabled, it must be a `varlen` function; otherwise, it must be a
            `batch` function.
    """
    if get_ring_attn_group() is not None:
        LOG.info("Ring attention already registered, exiting early...")
        return

    LOG.info(
        "Enabling ring attention sequence parallelism: "
        f"each sequence will be processed across {sequence_parallel_degree} GPUs"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

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
    elif ring_attn_func in [
        RingAttnFunc.BATCH_RING,
        RingAttnFunc.BATCH_ZIGZAG,
        RingAttnFunc.BATCH_STRIPE,
    ]:
        from axolotl.monkeypatch.attention.ring_attn.adapters.batch import (
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
