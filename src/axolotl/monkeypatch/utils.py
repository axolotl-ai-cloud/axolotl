"""
Shared utils for the monkeypatches
"""

import re
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils import is_torch_bf16_gpu_available


@torch.jit.script
def get_max_seqlen_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    max_num = int(torch.max(attention_mask).item())
    batch_size, _ = attention_mask.shape
    counts = torch.zeros((batch_size, max_num), dtype=torch.int32)
    for i in range(1, max_num + 1):
        mask = attention_mask == i
        counts[:, i - 1] = torch.sum(mask, dim=-1).to(dtype=torch.int32)
    result = counts.flatten()
    nonzero_indices = torch.nonzero(result).squeeze(-1)
    return result[nonzero_indices]


@torch.jit.script
def get_unpad_data(attention_mask: torch.Tensor):
    device = attention_mask.device
    seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = (
        F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        .to(device=device)
        .detach()
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_cu_seqlens(attn_mask):
    """generate a cumulative sequence length mask for flash attention using attn mask"""
    if len(attn_mask.shape) == 1:
        attn_mask = attn_mask.unsqueeze(0)

    device = attn_mask.device
    results = []
    max_seq_lens = []

    for row in attn_mask:
        # Exclude zeros to avoid adding their positions to the mask
        t_non_zeros = row[row != 0]
        # Find where the sequence number changes (including the first position)
        seq_change = torch.cat(
            [
                torch.tensor([1], dtype=torch.int32, device=device),
                t_non_zeros[1:] != t_non_zeros[:-1],
            ]
        )
        # Get the indices where the sequence changes
        change_indices = torch.cat(
            [
                (seq_change == 1).nonzero(as_tuple=True)[0],
                torch.tensor([len(t_non_zeros)], dtype=torch.int32, device=device),
            ]
        )
        # Calculate the sequence lengths
        seq_lengths = change_indices[1:] - change_indices[:-1]
        # Calculate the length of the final sequence or padding
        final_seq_length = len(row) - change_indices[-1]
        # Append the length of the final sequence or padding to seq_lengths
        if final_seq_length.item():
            seq_lengths = torch.cat(
                [
                    seq_lengths,
                    torch.tensor(
                        [final_seq_length.item()], dtype=torch.int32, device=device
                    ),
                ]
            )
        # Calculate the cumulative sequence lengths
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seq_lengths.cumsum(0)]
        )
        max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        results.append(cu_seqlens)
        max_seq_lens.append(max_seq_len)

    return torch.stack(results).to(dtype=torch.int32), torch.stack(max_seq_lens)


def get_cu_seqlens_from_pos_ids(
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """generate a cumulative sequence length mask for flash attention using pos ids"""
    if len(position_ids.shape) == 1:
        position_ids = position_ids.unsqueeze(0)

    device = position_ids.device
    results = []
    max_seq_lens = []

    for row in position_ids:
        # Count the number of consecutive zeros from the right side
        padding_length = (row == 0).int().flip(dims=[0]).cumprod(dim=0).sum().item()

        # Adjust the row to exclude padding
        adjusted_row = row[:-padding_length] if padding_length else row.clone()

        # Find where the position resets to 0 (indicating a new sequence)
        seq_starts = torch.cat(
            [
                torch.tensor([True], dtype=torch.bool, device=device),
                adjusted_row[1:] == 0,
            ]
        )
        # Get the indices where the sequence starts
        start_indices = torch.cat(
            [
                torch.nonzero(seq_starts).unbind(dim=1)[0],
                torch.tensor([len(adjusted_row)], dtype=torch.int32, device=device),
            ]
        )
        # Calculate the sequence lengths
        seq_lengths = start_indices[1:] - start_indices[:-1]
        # Calculate the cumulative sequence lengths
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seq_lengths.cumsum(0)]
        )
        # Append the padding length to the cumulative sequence lengths
        if padding_length:
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([len(row)], dtype=torch.int32, device=device)]
            )
        max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        results.append(cu_seqlens)
        max_seq_lens.append(max_seq_len)

    # Find the maximum value across all tensors
    max_value = max(t.max() for t in results)

    # Find the length of the longest tensor
    max_length = max(t.size(0) for t in results)

    # Pad each tensor to the same length and collect them in a list
    padded_results = [
        F.pad(t, (0, max_length - t.size(0)), "constant", max_value) for t in results
    ]

    return torch.stack(padded_results).to(dtype=torch.int32), torch.stack(max_seq_lens)


def set_module_name(model, name, value):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, value)


def mask_2d_to_4d(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    This expansion handles packed sequences so that sequences share the same attention mask integer value
    when they attend to each other within that sequence.
    This expansion transforms the mask to lower triangular form to prevent future peeking.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    mask = mask.unsqueeze(1).unsqueeze(2)
    mask = mask.expand(bsz, 1, tgt_len, src_len)

    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    binary_mask = torch.where(
        mask != 0,
        torch.tensor(1, device=mask.device).to(dtype),
        torch.tensor(0, device=mask.device).to(dtype),
    )

    # Create a block-diagonal mask.
    # we multiply by the binary mask so that 0's in the original mask are correctly excluded
    zero_one_mask = torch.eq(mask, mask.transpose(-1, -2)).int() * binary_mask

    # Now let's create a lower triangular mask of ones that will zero out the upper triangular part
    lower_triangular_ones = torch.tril(torch.ones((tgt_len, src_len), dtype=dtype)).to(
        mask.device
    )

    # Use the lower triangular mask to zero out the upper triangular part of the zero_one_mask
    masked_zero_one_mask = zero_one_mask * lower_triangular_ones

    return masked_zero_one_mask


def patched_prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *args,
):
    dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float32
    return _prepare_4d_causal_attention_mask(
        mask_2d_to_4d(attention_mask, dtype=dtype),
        *args,
    )


def patched_prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    *args,
):
    dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float32
    return _prepare_4d_causal_attention_mask_for_sdpa(
        mask_2d_to_4d(attention_mask, dtype=dtype),
        *args,
    )


def detab_code(code: str) -> Tuple[str, str]:
    try:
        spaces = re.match(r"([\s\t]{1,})", code).group(0)
        code = re.sub(r"^" + spaces, "", code, flags=re.MULTILINE)
    except AttributeError:
        return code, ""
    return code, spaces
