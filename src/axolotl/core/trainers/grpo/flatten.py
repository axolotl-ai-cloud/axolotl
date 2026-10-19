"""Host-side bookkeeping for the padding-free (batch flattening) GRPO forward."""

import torch


def chunk_lengths(
    chunk_mask: torch.Tensor,
    prompt_chunk_mask: torch.Tensor | None,
    logits_to_keep: int,
) -> tuple[list[int], list[int], list[int]]:
    """Per-sequence total length, prompt length and cumulative offsets as Python ints.

    One host sync for the whole chunk instead of one per sequence.
    """
    seq_lens = chunk_mask.sum(dim=1)
    if prompt_chunk_mask is not None:
        lens, plens = torch.stack([seq_lens, prompt_chunk_mask.sum(dim=1)]).tolist()
    else:
        lens = seq_lens.tolist()
        plens = [max(1, slen - logits_to_keep) for slen in lens]
    cu = [0]
    for slen in lens:
        cu.append(cu[-1] + slen)
    return [int(x) for x in lens], [int(x) for x in plens], cu


def completion_gather_index(
    lens: list[int],
    plens: list[int],
    cu: list[int],
    logits_to_keep: int,
    row_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index of every completion token's shifted logit position in the flat chunk.

    Returns CPU long tensors ``(rows, cols, src)`` such that
    ``out[rows, cols] = flat[src]`` scatters the flat per-position values into a
    ``(B, logits_to_keep)`` layout. Position ``cu[i] + plen - 1`` of the shifted
    flat tensor predicts sequence ``i``'s first completion token.
    """
    total = cu[-1]
    starts, counts = [], []
    for i, (slen, plen) in enumerate(zip(lens, plens, strict=True)):
        start = max(0, cu[i] + plen - 1)
        actual = min(slen - plen, logits_to_keep, total - 1 - start)
        starts.append(start)
        counts.append(max(actual, 0))
    counts_t = torch.tensor(counts, dtype=torch.long)
    rows = torch.repeat_interleave(
        torch.arange(row_offset, row_offset + len(lens)), counts_t
    )
    offsets = torch.repeat_interleave(torch.cumsum(counts_t, 0) - counts_t, counts_t)
    cols = torch.arange(int(counts_t.sum())) - offsets
    src = cols + torch.repeat_interleave(
        torch.tensor(starts, dtype=torch.long), counts_t
    )
    return rows, cols, src


def boundary_index(cu: list[int]) -> torch.Tensor:
    """Shifted-flat positions where sequence ``i-1``'s last token predicts sequence ``i``'s first."""
    limit = cu[-1] - 1
    return torch.tensor(
        [c - 1 for c in cu[1:-1] if 0 <= c - 1 < limit], dtype=torch.long
    )


def to_device_async(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host-to-device copy that does not drain the stream (pinned, non-blocking on CUDA)."""
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)
