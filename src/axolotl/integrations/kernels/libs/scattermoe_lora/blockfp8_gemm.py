"""Triton blockwise-fp8 (128x128) contiguous-grouped GEMM — native fp8 base for the MoE experts.

Reads the checkpoint's ``float8_e4m3fn`` weight + ``weight_scale_inv`` [E, N/128, K/128] directly:
upcasts each 128x128 block to bf16 in-register and applies its scale during accumulation (W8A16-fp8
= fp8 weight x bf16 act -> half the weight read+write bandwidth, no dequant materialization, no
requant). Arch-portable Triton, so it runs (and is verifiable) on sm120, unlike DeepGEMM's fp8xfp8
grouped kernel (sm90/sm100 only). Single deterministic config -> no runtime-autotune desync.

Layout matches the scattermoe grouped path: rows are TILE-padded and contiguous-grouped by expert;
``m_indices[i]`` is the expert id of the i-th ``BLK_M``(=tile)-row block.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLK = 128  # weight block (128x128); one scale per (n-block, k-block) tile


@triton.jit
def _blockfp8_grouped_fwd_kernel(
    A,  # [Mt, K] bf16 activations (contiguous-grouped, tile-padded)
    W,  # [E, N, K] float8_e4m3fn weight
    WS,  # [E, N/128, K/128] fp32 block scale_inv
    OUT,  # [Mt, N] bf16
    MIDX,  # [Mt/BLK_M] int32 — expert id per BLK_M-row block
    M,
    N,
    K,
    sa0: tl.constexpr,
    sa1: tl.constexpr,
    sw0,
    sw1: tl.constexpr,
    sw2: tl.constexpr,
    sws0,
    sws1: tl.constexpr,
    sws2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row (token) block
    pid_n = tl.program_id(1)  # N block (== weight block row, BLK=128)
    e = tl.load(MIDX + pid_m).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M)
    rn = pid_n * BLK + tl.arange(0, BLK)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLK_M, BLK), tl.float32)
    n_kblocks = tl.cdiv(K, BLK)
    for kb in range(n_kblocks):
        rk = kb * BLK + tl.arange(0, BLK)
        k_mask = rk < K
        # A tile [BLK_M, BLK] (bf16)
        a = tl.load(
            A + rm[:, None] * sa0 + rk[None, :] * sa1,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # W tile [BLK(k), BLK(n)] read as fp8 -> bf16 (b operand for tl.dot: [K, N])
        w = tl.load(
            W + e * sw0 + rn[None, :] * sw1 + rk[:, None] * sw2,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        # one 128x128 block scale for this (n-block, k-block) tile
        sc = tl.load(WS + e * sws0 + pid_n * sws1 + kb * sws2).to(tl.float32)
        acc += tl.dot(a, w) * sc
    tl.store(
        OUT + rm[:, None] * so0 + rn[None, :] * so1,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _blockfp8_grouped_dx_kernel(
    G,  # [Mt, N] bf16 grad-out
    W,  # [E, N, Kout] float8_e4m3fn weight (Kout = forward in-features)
    WS,  # [E, N/128, Kout/128] fp32 block scale_inv
    OUT,  # [Mt, Kout] bf16 = G @ W (per group)
    MIDX,  # [Mt/BLK_M] int32 expert id per row block
    M,
    N,
    KO,
    sg0: tl.constexpr,
    sg1: tl.constexpr,
    sw0,
    sw1: tl.constexpr,
    sw2: tl.constexpr,
    sws0,
    sws1: tl.constexpr,
    sws2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row (token) block
    pid_ko = tl.program_id(1)  # output-K block (== weight block col)
    e = tl.load(MIDX + pid_m).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M)
    rko = pid_ko * BLK + tl.arange(0, BLK)
    m_mask = rm < M
    ko_mask = rko < KO
    acc = tl.zeros((BLK_M, BLK), tl.float32)
    n_nblocks = tl.cdiv(N, BLK)
    for nb in range(n_nblocks):
        rn = nb * BLK + tl.arange(0, BLK)
        n_mask = rn < N
        g = tl.load(
            G + rm[:, None] * sg0 + rn[None, :] * sg1,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        # W tile [BLK(n), BLK(ko)] read fp8 -> bf16 (b operand: [N, KO])
        w = tl.load(
            W + e * sw0 + rn[:, None] * sw1 + rko[None, :] * sw2,
            mask=n_mask[:, None] & ko_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        sc = tl.load(WS + e * sws0 + nb * sws1 + pid_ko * sws2).to(tl.float32)
        acc += tl.dot(g, w) * sc
    tl.store(
        OUT + rm[:, None] * so0 + rko[None, :] * so1,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & ko_mask[None, :],
    )


def blockfp8_grouped_dx(
    g: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    m_indices: torch.Tensor,
    tile: int,
) -> torch.Tensor:
    """dX = ``g @ W`` (contract N) reading block-fp8 W in-register — the backward companion of
    :func:`blockfp8_grouped_gemm` (no bf16 weight materialization). ``g`` [Mt, N] bf16;
    ``wq`` [E, N, Kout] e4m3; ``ws`` [E, N/128, Kout/128] fp32 -> [Mt, Kout] bf16."""
    Mt, N = g.shape
    E, _, KO = wq.shape
    out = torch.empty(Mt, KO, device=g.device, dtype=torch.bfloat16)
    grid = (Mt // tile, triton.cdiv(KO, _BLK))
    _blockfp8_grouped_dx_kernel[grid](
        g,
        wq,
        ws,
        out,
        m_indices,
        Mt,
        N,
        KO,
        g.stride(0),
        g.stride(1),
        wq.stride(0),
        wq.stride(1),
        wq.stride(2),
        ws.stride(0),
        ws.stride(1),
        ws.stride(2),
        out.stride(0),
        out.stride(1),
        BLK_M=tile,
        BLK=_BLK,
        num_warps=4,
        num_stages=3,
    )
    return out


def blockfp8_grouped_gemm(
    a_bf16: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    m_indices: torch.Tensor,
    tile: int,
) -> torch.Tensor:
    """Grouped ``A @ W^T`` reading block-fp8 W in-register. ``a_bf16`` [Mt, K] bf16 (tile-padded,
    contiguous-grouped); ``wq`` [E, N, K] e4m3; ``ws`` [E, N/128, K/128] fp32; ``m_indices``
    [Mt/tile] int32 expert-id per tile -> [Mt, N] bf16. N and K need not be 128-aligned (masked)."""
    Mt, K = a_bf16.shape
    E, N, _ = wq.shape
    out = torch.empty(Mt, N, device=a_bf16.device, dtype=torch.bfloat16)
    grid = (Mt // tile, triton.cdiv(N, _BLK))
    _blockfp8_grouped_fwd_kernel[grid](
        a_bf16,
        wq,
        ws,
        out,
        m_indices,
        Mt,
        N,
        K,
        a_bf16.stride(0),
        a_bf16.stride(1),
        wq.stride(0),
        wq.stride(1),
        wq.stride(2),
        ws.stride(0),
        ws.stride(1),
        ws.stride(2),
        out.stride(0),
        out.stride(1),
        BLK_M=tile,
        BLK=_BLK,
        num_warps=4,
        num_stages=3,
    )
    return out
