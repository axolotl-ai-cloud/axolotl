"""Helper KD utils"""

import math
from typing import List, Union

import numpy as np
import torch
from torch import FloatTensor, Tensor


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


def strided_chunk_views(
    tensor: Union[np.ndarray, torch.Tensor],
    chunks: int,
    dim: int = 0,
    stride: int = 1,
    chunk_size: int | None = None,
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Split a tensor into chunks along a dimension with striding, prioritizing views over copies.

    Args:
        tensor: Input tensor (numpy array or torch tensor)
        chunks: Number of chunks to create
        dim: Dimension along which to chunk (default: 0)
        stride: Stride between chunk starting positions (default: 1)
        chunk_size: Size of each chunk. If None, calculated automatically (default: None)

    Returns:
        List of tensor chunks (views when possible, copies when necessary)
    """

    # Get the size of the specified dimension
    dim_size = tensor.shape[dim]

    # Calculate chunk size if not provided
    if chunk_size is None:
        chunk_size = (dim_size + chunks - 1) // chunks  # Ceiling division

    chunks_list = []

    for i in range(chunks):
        start_idx = i * stride
        end_idx = min(start_idx + chunk_size, dim_size)

        # Break if we've gone beyond the tensor
        if start_idx >= dim_size:
            break

        # Create slice objects for all dimensions
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start_idx, end_idx)

        chunk = tensor[tuple(slices)]
        chunks_list.append(chunk)

    return chunks_list


def chunk_overlap(input_tensor: Tensor, chunks: int, dim: int = 0, overlap: int = 1):
    dim_size = input_tensor.shape[dim]
    stride = math.ceil(dim_size / chunks)

    return strided_chunk_views(
        input_tensor, chunks, dim, stride=stride, chunk_size=stride + overlap
    )
