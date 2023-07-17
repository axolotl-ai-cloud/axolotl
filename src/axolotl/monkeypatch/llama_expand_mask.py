"""
expands the binary attention mask per 3.2.2 of https://arxiv.org/pdf/2107.02027.pdf
"""
from typing import Optional

import torch


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Initialize a tensor to hold the expanded masks
    expanded_masks = torch.zeros(bsz, 1, tgt_len, src_len).to(dtype)

    # For each sequence in the batch
    for i in range(bsz):
        # Get the mask for this sequence
        mask_i = mask[i].unsqueeze(0)

        # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
        binary_mask_i = torch.where(
            mask_i != 0,
            torch.tensor(1).to(dtype),
            torch.tensor(0).to(dtype),
        )

        # Create a block-diagonal mask
        zero_one_mask_i = torch.eq(mask_i, mask_i.t()).int() * binary_mask_i

        # Expand the mask
        expanded_mask_i = zero_one_mask_i.unsqueeze(0).expand(1, 1, tgt_len, src_len)

        # Store the expanded mask
        expanded_masks[i] = expanded_mask_i

    inverted_mask = 1.0 - expanded_masks

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def hijack_expand_mask():
    import transformers

    transformers.models.llama.modeling_llama._expand_mask = (  # pylint: disable=protected-access
        _expand_mask
    )
