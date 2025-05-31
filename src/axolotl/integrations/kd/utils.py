"""Helper KD utils"""

import torch
from torch import FloatTensor


def normalize_logprobs(logprobs: FloatTensor, topk: int) -> FloatTensor:
    """
    Re-normalizes top-k raw logprobs as probabilities, and converts back to logprobs.
    """
    # Ensure raw_logprobs matches kd_online_topk length for tensor operations
    # This should ideally be handled by the caller ensuring correct padding/truncation first
    if logprobs.shape[-1] != topk:
        # pad last dimension of logprobs to match topk length with -inf
        padding_len = topk - logprobs.shape[-1]
        padding_tensor = torch.full(
            (
                *logprobs.shape[:-1],
                padding_len,
            ),  # Takes all dimensions of logprobs except the last, then appends padding_needed
            float("-inf"),
            dtype=logprobs.dtype,
            device=logprobs.device,
        )
        logprobs = torch.cat((logprobs, padding_tensor), dim=-1)

    # Convert logprobs at T_online to probabilities
    # use log sum exp trick to avoid underflow
    position_logprobs_lse = torch.logsumexp(logprobs, dim=-1, keepdim=True)
    teacher_probs_t_online = torch.exp(logprobs - position_logprobs_lse)

    # Normalize probabilities (sum to 1)
    # This is important if the top-k from server aren't a full distribution
    teacher_probs_t_online_sum = teacher_probs_t_online.sum(dim=-1, keepdim=True)
    teacher_probs_t_online = teacher_probs_t_online / teacher_probs_t_online_sum

    final_logprobs_tensor = torch.log(teacher_probs_t_online)

    return final_logprobs_tensor
