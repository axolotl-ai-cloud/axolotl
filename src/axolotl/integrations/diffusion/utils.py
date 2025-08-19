import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers.masking_utils import find_packed_sequence_indices, packed_sequence_mask_function


def create_bidirectional_block_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> "BlockMask":
    """
    Creates a bidirectional block mask for FlexAttention.

    Args:
        input_ids: Input token ids [batch_size, seq_len]
        attention_mask: Padding mask [batch_size, seq_len]

    Returns:
        BlockMask for bidirectional attention with padding
    """
    batch_size, seq_len = input_ids.shape

    if position_ids is not None:
        packed_seq_mask = find_packed_sequence_indices(position_ids)
        mask_fn =packed_sequence_mask_function(packed_seq_mask, batch_size, seq_len)
    elif attention_mask is None:
        # If no padding mask, all positions can attend to all positions
        def mask_fn(b, h, q_idx, kv_idx):
            # Always return True for bidirectional attention
            return True
    else:
        # Convert attention_mask to boolean if needed
        attention_mask = attention_mask.bool()

        def mask_fn(b, h, q_idx, kv_idx):
            # Both query and key positions must be valid (not padding)
            return attention_mask[b, q_idx] & attention_mask[b, kv_idx]

    # Create the block mask
    block_mask = create_block_mask(
        mask_fn,
        B=batch_size,
        H=None,  # Will be set by the attention layer
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=input_ids.device,
        _compile=True,
    )

    return block_mask
