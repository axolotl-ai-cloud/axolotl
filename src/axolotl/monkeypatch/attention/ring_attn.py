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
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from axolotl.logging_config import configure_logging

try:
    from ring_flash_attn import update_ring_flash_attn_params
except ImportError:
    # We pass silently here, but raise an ImportError in our Axolotl config validation
    # if cfg.sequence_parallel_degree > 1 and `ring-flash-attn` is not installed.
    pass


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
        ring_attn_group: Process group for ring attention.
    """
    global RING_ATTN_GROUP  # pylint: disable=global-statement
    RING_ATTN_GROUP = ring_attn_group


def patch_flash_attention_for_sequential_batch(sequence_parallel_degree: int):
    """
    Patch flash attention a second time to handle batched data. This is a hack to
    accommodate certain RL trainers which batch data even when `micro_batch_size: 1` is
    specified in the Axolotl config.
    
    Args:
        sequence_parallel_degree: Sequence parallelism factor.
    """
    # Store the original flash attention function
    original_flash_attention = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    def sequential_batch_flash_attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        softcap: float | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        # Check if we have a batch dimension > 1
        batch_size = query.shape[0]
        
        if batch_size <= 1:
            return original_flash_attention(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout,
                scaling,
                sliding_window,
                softcap,
                **kwargs
            )
        
        # Process each item in the batch separately
        outputs = []
        
        for i in range(batch_size):
            # Extract single batch item
            q_item = query[i:i+1]
            k_item = key[i:i+1]
            v_item = value[i:i+1]
            
            # Handle attention mask - it might be None or have different shapes
            mask_item = None
            if attention_mask is not None:
                # The mask could have different formats depending on implementation
                if attention_mask.dim() >= 3 and attention_mask.shape[0] == batch_size:
                    mask_item = attention_mask[i:i+1]
                else:
                    # For broadcast masks that don't have a batch dimension
                    mask_item = attention_mask
            
            # At this point, inputs should already be partitioned by the sequence
            # parallel data collator
            batch_size = q_item.shape[0]
            seq_len = q_item.shape[2]
            packed_seq_lens = [seq_len] * batch_size

            # Calculate the full sequence length across all GPUs in this SP group
            total_seq_len = seq_len * sequence_parallel_degree

            cu_seqlens = torch.cumsum(
                torch.tensor(
                    packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32
                ),
                dim=-1,
                dtype=torch.int32,
            )
            cu_seqlens = F.pad(
                F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len
            )

            update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
            
            # Call the original function for a single batch item
            output, _ = original_flash_attention(
                module,
                q_item,
                k_item,
                v_item,
                mask_item,
                dropout,
                scaling,
                sliding_window,
                softcap,
                **kwargs
            )
            
            outputs.append(output)
            
            dist.barrier()
        
        # Concatenate results along batch dimension
        concatenated_output = torch.cat(outputs, dim=0)
        return concatenated_output, None
    
    # Replace the original function with our sequential version
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = sequential_batch_flash_attention


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
    patch_flash_attention_for_sequential_batch(sequence_parallel_degree)
