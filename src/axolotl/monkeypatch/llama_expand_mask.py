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

    binary_mask = torch.where(
        mask != 0, torch.tensor(1).to(torch.int16), torch.tensor(0).to(torch.int16)
    )

    zero_one_mask = torch.eq(mask, mask.t()).int() * binary_mask
    expanded_mask = zero_one_mask.unsqueeze(0).expand(bsz, 1, tgt_len, src_len)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def hijack_expand_mask():
    import transformers

    transformers.models.llama.modeling_llama._expand_mask = (  # pylint: disable=protected-access
        _expand_mask
    )
