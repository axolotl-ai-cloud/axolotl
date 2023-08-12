"""
Shared utils for the monkeypatches
"""
import torch


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


def get_cu_seqlens_from_pos_ids(position_ids):
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
                (seq_starts).nonzero(as_tuple=True)[0],
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

    return torch.stack(results).to(dtype=torch.int32), torch.stack(max_seq_lens)
