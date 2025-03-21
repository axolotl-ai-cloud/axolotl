"""
expands the binary attention mask per 3.2.2 of https://arxiv.org/pdf/2107.02027.pdf
"""

from typing import Optional

import torch

from axolotl.monkeypatch.utils import mask_2d_to_4d


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    masked_zero_one_mask = mask_2d_to_4d(mask, dtype, tgt_len)
    inverted_mask = 1.0 - masked_zero_one_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def hijack_expand_mask():
    import transformers

    transformers.models.llama.modeling_llama._expand_mask = (  # pylint: disable=protected-access
        _expand_mask
    )
