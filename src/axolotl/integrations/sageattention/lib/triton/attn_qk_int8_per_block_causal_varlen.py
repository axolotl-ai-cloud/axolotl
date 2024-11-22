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

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    kv_len,
    K_ptrs,
    K_scale_ptr,
    V_ptrs,
    stride_kn,
    stride_vn,
    start_m,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += (lo // BLOCK_N) * H
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += H
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    cu_seqlens_q,
    cu_seqlens_k,
    Q_scale,
    K_scale,
    cu_seqlens_q_scale,
    cu_seqlens_k_scale,
    Out,
    stride_qh,
    stride_qn,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_oh,
    stride_on,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

    qo_len = cu_seqlens_q_end - cu_seqlens_q_start

    if (start_m * BLOCK_M) >= qo_len:
        return

    cu_seq_lens_q_scale_start = tl.load(cu_seqlens_q_scale + off_z)
    cu_seq_lens_k_scale_start = tl.load(cu_seqlens_k_scale + off_z)

    q_scale_offset = cu_seq_lens_q_scale_start * H + off_h + start_m * H
    k_scale_offset = (
        cu_seq_lens_k_scale_start * (H // num_kv_groups) + off_h // num_kv_groups
    )

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)

    kv_len = cu_seqlens_k_end - cu_seqlens_k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = (
        Q
        + (cu_seqlens_q_start * stride_qn + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    Q_scale_ptr = Q_scale + q_scale_offset
    K_ptrs = (
        K
        + (cu_seqlens_k_start * stride_kn + (off_h // num_kv_groups) * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = (
        V
        + (cu_seqlens_k_start * stride_vn + (off_h // num_kv_groups) * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    O_block_ptr = (
        Out
        + (cu_seqlens_q_start * stride_on + off_h * stride_oh)
        + offs_m[:, None] * stride_on
        + offs_k[None, :]
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        q_scale,
        kv_len,
        K_ptrs,
        K_scale_ptr,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        H // num_kv_groups,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        4 - STAGE,
        offs_m,
        offs_n,
    )

    acc, l_i, _ = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        q_scale,
        kv_len,
        K_ptrs,
        K_scale_ptr,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        H // num_kv_groups,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        2,
        offs_m,
        offs_n,
    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))


@triton.jit
def _attn_bwd_inner(
    dq_acc,
    dk_acc,
    dv_acc,
    l_i,
    m_i,
    q,
    k,
    v,
    do,
    q_scale,
    k_scale,
    kv_len,
    stride_kn,
    stride_vn,
    start_m,
    H,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        k += stride_kn * lo
        v += stride_vn * lo

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k_curr = tl.load(k, mask=k_mask)
        v_curr = tl.load(v, mask=k_mask)
        k_scale_curr = tl.load(k_scale)
        s = tl.dot(q, k_curr, trans_b=True).to(tl.float32) * q_scale * k_scale_curr

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            s = s + tl.where(mask, 0.0, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(s, 1))
            s = s - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(s, 1))
            s = s - m_ij[:, None]

        p = tl.math.exp2(s)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        p = p / l_i[:, None]  # Normalize probabilities

        # Compute gradients
        # Compute softmax gradient
        do_scaled = do / l_i[:, None]
        dv_contrib = tl.dot(p.to(tl.float16).T, do_scaled.to(tl.float16))
        dv_acc += dv_contrib

        dp = tl.dot(do_scaled.to(tl.float16), v_curr.to(tl.float16).T)

        # Compute ds (gradient w.r.t. logits s)
        p_dp = p * dp
        sum_p_dp = tl.sum(p_dp, axis=1)
        ds = (p_dp - p * sum_p_dp[:, None]) * tl.math.log(2.0)  # Adjust for exp2

        # Compute gradients w.r.t q and k
        dq_contrib = tl.dot(ds.to(tl.float16), k_curr.to(tl.float16))
        dk_contrib = tl.dot(ds.to(tl.float16).T, q.to(tl.float16))

        dq_acc += dq_contrib * (q_scale * k_scale_curr)
        dk_acc += dk_contrib * (q_scale * k_scale_curr)

        k += BLOCK_N * stride_kn
        k_scale += H
        v += BLOCK_N * stride_vn

    return dq_acc, dk_acc, dv_acc, l_i, m_i


@triton.jit
def _attn_bwd(
    DO,
    Q,
    K,
    V,
    cu_seqlens_q,
    cu_seqlens_k,
    Q_scale,
    K_scale,
    cu_seqlens_q_scale,
    cu_seqlens_k_scale,
    L,
    M,
    DQ,
    DK,
    DV,
    stride_qh,
    stride_qn,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    qo_len = cu_seqlens_q_end - cu_seqlens_q_start

    if (start_m * BLOCK_M) >= qo_len:
        return

    cu_seq_lens_q_scale_start = tl.load(cu_seqlens_q_scale + off_z)
    cu_seq_lens_k_scale_start = tl.load(cu_seqlens_k_scale + off_z)

    q_scale_offset = cu_seq_lens_q_scale_start * H + off_h + start_m * H
    k_scale_offset = (
        cu_seq_lens_k_scale_start * (H // num_kv_groups) + off_h // num_kv_groups
    )

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
    kv_len = cu_seqlens_k_end - cu_seqlens_k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = (
        Q
        + (cu_seqlens_q_start * stride_qn + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    DO_ptrs = (
        DO
        + (cu_seqlens_q_start * stride_qn + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    Q_scale_ptr = Q_scale + q_scale_offset
    K_ptrs = (
        K
        + (cu_seqlens_k_start * stride_kn + (off_h // num_kv_groups) * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = (
        V
        + (cu_seqlens_k_start * stride_vn + (off_h // num_kv_groups) * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    DQ_ptrs = (
        DQ
        + (cu_seqlens_q_start * stride_qn + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    DK_ptrs = (
        DK
        + (cu_seqlens_k_start * stride_kn + (off_h // num_kv_groups) * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    DV_ptrs = (
        DV
        + (cu_seqlens_k_start * stride_vn + (off_h // num_kv_groups) * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    L_ptrs = L + (cu_seqlens_q_start + offs_m)
    M_ptrs = M + (cu_seqlens_q_start + offs_m)

    m_i = tl.load(M_ptrs, mask=offs_m < qo_len, other=float("-inf"))
    l_i = tl.load(L_ptrs, mask=offs_m < qo_len, other=1.0)

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    do = tl.load(DO_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)

    dq_acc, dk_acc, dv_acc, l_i, m_i = _attn_bwd_inner(
        dq_acc,
        dk_acc,
        dv_acc,
        l_i,
        m_i,
        q,
        K_ptrs,
        V_ptrs,
        do,
        q_scale,
        K_scale_ptr,
        kv_len,
        stride_kn,
        stride_vn,
        start_m,
        H // num_kv_groups,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        4 - STAGE,
        offs_m,
        offs_n,
    )

    dq_acc, dk_acc, dv_acc, l_i, m_i = _attn_bwd_inner(
        dq_acc,
        dk_acc,
        dv_acc,
        l_i,
        m_i,
        q,
        K_ptrs,
        V_ptrs,
        do,
        q_scale,
        K_scale_ptr,
        kv_len,
        stride_kn,
        stride_vn,
        start_m,
        H // num_kv_groups,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        2,
        offs_m,
        offs_n,
    )

    tl.store(DQ_ptrs, dq_acc.to(DQ.dtype.element_ty), mask=offs_m[:, None] < qo_len)
    tl.store(DK_ptrs, dk_acc.to(DK.dtype.element_ty), mask=offs_n[None, :] < kv_len)
    tl.store(DV_ptrs, dv_acc.to(DV.dtype.element_ty), mask=offs_n[:, None] < kv_len)


def forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    q_scale,
    k_scale,
    cu_seqlens_q_scale,
    cu_seqlens_k_scale,
    output_dtype=torch.float16,
):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    b = cu_seqlens_q.shape[0] - 1
    _, h_qo, head_dim = q.shape
    _, h_kv, _ = k.shape

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        q_scale,
        k_scale,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
        o,
        q.stride(1),
        q.stride(0),
        k.stride(1),
        k.stride(0),
        v.stride(1),
        v.stride(0),
        o.stride(1),
        o.stride(0),
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
    )
    return o


def backward(
    do,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    q_scale,
    k_scale,
    cu_seqlens_q_scale,
    cu_seqlens_k_scale,
    l,
    m,
    output_dtype=torch.float16,
):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

    device = q.device
    dtype = q.dtype
    b = cu_seqlens_q.shape[0] - 1
    _, h_qo, head_dim = q.shape
    _, h_kv, _ = k.shape
    num_kv_groups = h_qo // h_kv

    dq = torch.zeros_like(q, dtype=output_dtype)
    dk = torch.zeros_like(k, dtype=output_dtype)
    dv = torch.zeros_like(v, dtype=output_dtype)

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), h_qo, b)
    _attn_bwd[grid](
        do,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        q_scale,
        k_scale,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
        l,
        m,
        dq,
        dk,
        dv,
        q.stride(1),
        q.stride(0),
        k.stride(1),
        k.stride(0),
        v.stride(1),
        v.stride(0),
        h_qo,
        num_kv_groups,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=stage,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
    )
    return dq, dk, dv


# class TritonAttentionFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale):
#         l = torch.zeros(q.shape[0], device=q.device, dtype=torch.float32)
#         m = torch.zeros(q.shape[0], device=q.device, dtype=torch.float32)
#         output = forward(q, k, v, cu_seqlens_q, cu_seqlens_k, q.shape[0], q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, l, m)
#         ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, l, m)
#         return output
#
#     @staticmethod
#     def backward(ctx, do):
#         q, k, v, cu_seqlens_q, cu_seqlens_k, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, l, m = ctx.saved_tensors
#         dq, dk, dv = backward(
#             do, q, k, v,
#             cu_seqlens_q, cu_seqlens_k,
#             q.shape[0], q_scale, k_scale,
#             cu_seqlens_q_scale, cu_seqlens_k_scale,
#             l, m,
#         )
#         return dq, dk, dv, None, None, None, None, None, None
