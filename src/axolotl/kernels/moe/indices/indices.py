"""Vendored token permutation kernels from TorchTitan."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

__all__ = ["generate_permute_indices"]


@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for expert_id in range(pid, experts_per_rank, num_programs):
        write_offset = tl.load(write_offsets_ptr + expert_id)

        for r in range(num_ranks):
            idx = r * experts_per_rank + expert_id

            start_index = tl.load(start_index_values_ptr + idx)
            length = tl.load(tokens_per_expert_group_ptr + idx)

            offsets = tl.arange(0, BLOCK_SIZE)

            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets
                mask = chunk_offsets < length
                values = start_index + chunk_offsets
                dest_indices = write_offset + chunk_offsets
                tl.store(output_ptr + dest_indices, values, mask=mask)

            write_offset += length


def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
):
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )
    num_blocks = min(experts_per_rank, max_blocks)
    grid = (num_blocks,)
    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32)
    for expert_id in range(experts_per_rank):
        write_start = write_offsets[expert_id].item()
        for r in range(num_ranks):
            idx = r * experts_per_rank + expert_id
            start_index = start_index_values[idx].item()
            length = tokens_per_expert_group[idx].item()
            if length > 0:
                end_idx = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                )
            write_start += length
    return permuted_indices


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    use_cpu: bool = False,
):
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)
