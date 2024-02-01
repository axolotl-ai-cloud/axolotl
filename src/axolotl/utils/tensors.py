import torch
import torch.nn.functional as F


def keep_unpacked_data(data: torch.Tensor, index=None, nonzero_total=None, pad_val= None, pairs=False):
    # pad val could be padding token (input_ids), -100 (labels), or 0 (attention_mask)
    if index >= nonzero_total:
        return False
    if pairs and (index // 2) >= (nonzero_total // 2):
        return False
    if pad_val and (data == pad_val).all(dim=0).all():
        return False
    return True


def split_and_pad_packed(tensor, cu_seqlens, max_seqlen, keep_fn=None):
    split_tensors = []

    counts = count_nonzero_sequences(cu_seqlens)
    # Iterate over each batch
    for i in range(tensor.size(0)):
        seq_lens = cu_seqlens[i]
        start_idx = 0

        # Iterate over the cumulative sequence lengths
        for j, end_idx in enumerate(seq_lens[1:]):
            if end_idx == start_idx:
                break
            # Extract and pad the current sequence
            current_seq = tensor[i, start_idx:end_idx]
            keep = True
            if keep_fn:
                keep = keep_fn(current_seq, index=j, nonzero_total=counts[i])
            if not keep:
                continue
            padding_size = max_seqlen - current_seq.size(0)
            padded_seq = F.pad(current_seq, (0, 0) * (current_seq.dim() - 2) + (0, padding_size))

            # Append the padded sequence to the list
            split_tensors.append(padded_seq)

            # Update start index for the next sequence
            start_idx = end_idx

    # Stack the padded tensors
    return torch.stack(split_tensors, dim=0)


def count_nonzero_sequences(cu_seqlens: torch.Tensor) -> torch.LongTensor:
    diffs = torch.diff(cu_seqlens, dim=1, prepend=torch.zeros(cu_seqlens.shape[0], 1, dtype=cu_seqlens.dtype))
    valid_lengths = diffs != 0
    counts = valid_lengths.sum(dim=1).long()

    return counts


# Example usage
# Example tensor with dimensions [batch_size, seq_len, other_dimensions...]
# example_tensor = torch.randn(batch_size, seq_len, other_dimensions...)
# cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(batch["position_ids"])
# split_padded_tensor = split_and_pad_packed(example_tensor, cu_seqlens, max_seqlen)
