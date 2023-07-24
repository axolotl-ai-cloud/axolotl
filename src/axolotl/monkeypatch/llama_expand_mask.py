"""
expands the binary attention mask per 3.2.2 of https://arxiv.org/pdf/2107.02027.pdf
"""
from typing import Optional

import torch


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    This expansion handles packed sequences so that sequences share the same attention mask integer value
    when they attend to each other within that sequence. This should result in a block diagonal mask
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    mask = mask.unsqueeze(1).unsqueeze(2)
    mask = mask.expand(bsz, 1, tgt_len, src_len)

    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    binary_mask = torch.where(
        mask != 0,
        torch.tensor(1).to(dtype),
        torch.tensor(0).to(dtype),
    )

    # Create a block-diagonal mask.
    # we multiply by the binary mask so that 0's in the original mask are correctly excluded
    zero_one_mask = torch.eq(mask, mask.transpose(-1, -2)).int() * binary_mask

    # Expand the mask to the correct dimensions for the current batch index
    expanded_mask = zero_one_mask.expand(bsz, 1, tgt_len, src_len)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def hijack_expand_mask():
    import transformers

    transformers.models.llama.modeling_llama._expand_mask = (  # pylint: disable=protected-access
        _expand_mask
    )
