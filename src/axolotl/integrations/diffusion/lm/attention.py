"""Packed full-sequence FlexAttention mask for native diffusion."""

from __future__ import annotations

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask
except ImportError:
    create_block_mask = None


_compiled_create_block_mask = (
    torch.compile(create_block_mask, dynamic=True)
    if create_block_mask is not None and hasattr(torch, "compile")
    else create_block_mask
)


def full_sequence_flex_block_mask(
    document_ids: torch.Tensor,
    semantic_validity: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None = None,
):
    """Create a bidirectional mask restricted to each logical document."""
    if _compiled_create_block_mask is None:
        raise RuntimeError(
            "flex_attention requires a PyTorch build with BlockMask support"
        )
    if sliding_window is not None:
        raise ValueError("full-sequence diffusion does not support sliding attention")
    del position_ids
    batch_size, length = document_ids.shape

    def mask_mod(batch, head, query, key):
        del head
        return (
            semantic_validity[batch, query]
            & semantic_validity[batch, key]
            & (document_ids[batch, query] >= 0)
            & (document_ids[batch, query] == document_ids[batch, key])
        )

    block_mask_factory = (
        _compiled_create_block_mask if document_ids.is_cuda else create_block_mask
    )
    return block_mask_factory(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device=document_ids.device,
    )
