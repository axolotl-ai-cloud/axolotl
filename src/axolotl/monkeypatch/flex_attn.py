"""
Taken from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention_utils.py
"""
from typing import Union

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import (
    create_block_mask as create_block_causal_mask_flex,
)

_MaskType = Union[torch.Tensor, BlockMask]


def create_block_causal_mask(
    seq_lens: list[torch.Tensor], max_seq_len: int
) -> torch.Tensor:
    """
    Given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.


    Returns:
        Tensor: Block causal mask of shape (batch_size, max_seq_len, max_seq_len).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.trilu( # torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq_len.device)
            )
            for seq_len in seq_lens[sample_idx]
        ]

        """residue_len = max_seq_len - torch.sum(seq_lens[sample_idx])
        block_attn_masks.append(
            torch.tril(
                torch.ones(
                    residue_len, residue_len, dtype=torch.bool, device=seq_lens[sample_idx].device
                )
            )
        )"""

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))

    return torch.stack(batch_block_attn_masks)[:, None, :, :]


def _get_document_ids_from_seq_lens(
    seq_lens: list[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
    For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        Tensor: Document IDs of shape (batch_size, max_seq_len).
    """
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        # We assume seq lens sum to max seq lens, so document_ids should be of
        # shape (max_seq_len, )
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=seq_len.device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    batch_document_ids = torch.stack(batch_document_ids)
    return batch_document_ids


def packed_block_causal_mask(
    seq_lens: list[torch.Tensor], totalseqlens: list[int]
) -> _MaskType:
    """
    Create a block causal document mask for a batch of packed sequences. If
    flex attention is supported by the current hardware, block causal logic and
    passing this into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. If on an older version, a standard 2D block causal mask is created and returned.

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """

    document_ids = _get_document_ids_from_seq_lens(seq_lens)
    batch_size , max_seq_len = document_ids
    document_ids = document_ids.to("cuda")

    # Instead of passing a tensor mask, flex attention requires a mask_mod function
    # that determines which elements of QK^T should be included in the attention
    # computation prior to the softmax. For sample packing, we need both the
    # logic for both causal mask and document mask. See PyTorch's official
    # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
    def mask_mod(b, h, q_idx, kv_idx):
        """
        Defines the logic of a block causal mask by combining both a standard causal mask
        and a block diagonal document mask.

        See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
        for an illustration.
        """
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask & (q_idx < totalseqlens[b])

    return create_block_causal_mask_flex(
        mask_mod,
        batch_size,
        None,
        max_seq_len,
        max_seq_len,
        device="cuda",
        BLOCK_SIZE=512,
    )
