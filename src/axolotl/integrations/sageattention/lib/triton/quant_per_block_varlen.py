"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    cu_seqlens_input,
    cu_seqlens_scale,
    stride_ih,
    stride_in,
    stride_oh,
    stride_on,
    sm_scale,
    H: tl.constexpr,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    cu_seqlens_input_start = tl.load(cu_seqlens_input + off_b)
    cu_seqlens_input_end = tl.load(cu_seqlens_input + off_b + 1)

    L = cu_seqlens_input_end - cu_seqlens_input_start

    if (off_blk * BLK) >= L:
        return

    cu_seqlens_scale_start = tl.load(cu_seqlens_scale + off_b)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = (
        Input
        + cu_seqlens_input_start * stride_in
        + off_h * stride_ih
        + offs_n[:, None] * stride_in
        + offs_k[None, :]
    )
    output_ptrs = (
        Output
        + cu_seqlens_input_start * stride_on
        + off_h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_k[None, :]
    )
    scale_ptrs = Scale + cu_seqlens_scale_start * H + off_h + off_blk * H

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8(
    q,
    k,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    h_qo = q.shape[1]
    h_kv = k.shape[1]
    head_dim = q.shape[-1]

    b = cu_seqlens_q.shape[0] - 1
    q_batch_len = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_batch_len = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    q_scale_len = (q_batch_len + BLKQ - 1) // BLKQ
    k_scale_len = (k_batch_len + BLKK - 1) // BLKK

    cu_seqlens_q_scale = torch.nn.functional.pad(
        torch.cumsum(q_scale_len, dim=0), (1, 0), value=0
    )
    cu_seqlens_k_scale = torch.nn.functional.pad(
        torch.cumsum(k_scale_len, dim=0), (1, 0), value=0
    )

    q_scale = torch.empty(
        (cu_seqlens_q_scale[-1], h_qo), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (cu_seqlens_k_scale[-1], h_kv), device=k.device, dtype=torch.float32
    )

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((max_seqlen_q + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        cu_seqlens_q,
        cu_seqlens_q_scale,
        q.stride(1),
        q.stride(0),
        q_int8.stride(1),
        q_int8.stride(0),
        sm_scale=(sm_scale * 1.44269504),
        H=h_qo,
        C=head_dim,
        BLK=BLKQ,
    )

    grid = ((max_seqlen_k + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k,
        k_int8,
        k_scale,
        cu_seqlens_k,
        cu_seqlens_k_scale,
        k.stride(1),
        k.stride(0),
        k_int8.stride(1),
        k_int8.stride(0),
        sm_scale=1.0,
        H=h_kv,
        C=head_dim,
        BLK=BLKK,
    )

    return q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale
