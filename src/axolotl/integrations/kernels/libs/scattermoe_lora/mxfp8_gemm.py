"""Triton OCP-MX fp8 (e4m3 data + e8m0 per-32-block scale) contiguous-grouped GEMM — native mxfp8
base for the MoE experts.

Reads an mxfp8 weight ``float8_e4m3fn`` [E, N, K] + e8m0 block scale [E, N, K/32] directly: upcasts
each e4m3 element to bf16 in-register and applies its 32-wide block scale during accumulation
(W8A16 = fp8 weight x bf16 act -> half the weight read+write bandwidth, no dequant materialization).
The MX block axis is the contraction axis K; the scale for output row ``n`` and K sub-block ``kb`` is
``2^(byte-127)`` (e8m0 stores the biased exponent). Arch-portable Triton, so it runs (and is
verifiable) on sm120. Single deterministic config -> no runtime-autotune desync.

Layout matches the scattermoe grouped path: rows are TILE-padded and contiguous-grouped by expert;
``m_indices[i]`` is the expert id of the i-th ``BLK_M``(=tile)-row block. The e8m0 scale is passed as
a ``uint8`` view (Triton loads the raw exponent byte).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MX = 32  # OCP-MX block size (one e8m0 scale per 32 elements along K)


@triton.jit
def _mxfp8_grouped_fwd_kernel(
    A,  # [Mt, K] bf16 activations (contiguous-grouped, tile-padded)
    W,  # [E, N, K] float8_e4m3fn weight
    WS,  # [E, N, K/32] uint8 (e8m0 biased-exponent byte)
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
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,  # == _MX (one scale vector per K step)
):
    pid_m = tl.program_id(0)  # row (token) block
    pid_n = tl.program_id(1)  # N (output-feature) block
    e = tl.load(MIDX + pid_m).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLK_M, BLK_N), tl.float32)
    n_ksteps = tl.cdiv(K, BLK_K)
    for ks in range(n_ksteps):
        rk = ks * BLK_K + tl.arange(0, BLK_K)
        k_mask = rk < K
        # A tile [BLK_M, BLK_K] bf16
        a = tl.load(
            A + rm[:, None] * sa0 + rk[None, :] * sa1,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # W tile [BLK_K, BLK_N] read e4m3 -> fp32 (b operand for tl.dot: [K, N])
        w = tl.load(
            W + e * sw0 + rn[None, :] * sw1 + rk[:, None] * sw2,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # one e8m0 scale per (n, K-block); this K step is a single 32-wide block -> vector over N
        s_e8 = tl.load(
            WS + e * sws0 + rn * sws1 + ks * sws2,
            mask=n_mask,
            other=0,
        ).to(tl.int32)
        sc = tl.exp2((s_e8 - 127).to(tl.float32))  # e8m0 byte -> 2^(E-127)
        w = (w * sc[None, :]).to(tl.bfloat16)
        acc += tl.dot(a, w)
    tl.store(
        OUT + rm[:, None] * so0 + rn[None, :] * so1,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _mxfp8_grouped_dx_kernel(
    G,  # [Mt, N] bf16 grad-out
    W,  # [E, N, Kout] float8_e4m3fn weight (Kout = forward in-features)
    WS,  # [E, N, Kout/32] uint8 (e8m0)
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
    BLK_N: tl.constexpr,  # contraction step over N (== _MX not required; N is the summed axis)
    BLK_KO: tl.constexpr,  # output-K block (== _MX: scale is per-32 along Kout)
):
    pid_m = tl.program_id(0)  # row (token) block
    pid_ko = tl.program_id(1)  # output-K block (one 32-wide MX block)
    e = tl.load(MIDX + pid_m).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M)
    rko = pid_ko * BLK_KO + tl.arange(0, BLK_KO)
    m_mask = rm < M
    ko_mask = rko < KO
    acc = tl.zeros((BLK_M, BLK_KO), tl.float32)
    n_nblocks = tl.cdiv(N, BLK_N)
    for nb in range(n_nblocks):
        rn = nb * BLK_N + tl.arange(0, BLK_N)
        n_mask = rn < N
        g = tl.load(
            G + rm[:, None] * sg0 + rn[None, :] * sg1,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        # W tile [BLK_N, BLK_KO] read e4m3 -> fp32 (b operand: [N, KO])
        w = tl.load(
            W + e * sw0 + rn[:, None] * sw1 + rko[None, :] * sw2,
            mask=n_mask[:, None] & ko_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # scale per (n, ko-block); this pid_ko is one 32-wide Kout block -> vector over N
        s_e8 = tl.load(
            WS + e * sws0 + rn * sws1 + pid_ko * sws2,
            mask=n_mask,
            other=0,
        ).to(tl.int32)
        sc = tl.exp2((s_e8 - 127).to(tl.float32))
        w = (w * sc[:, None]).to(tl.bfloat16)
        acc += tl.dot(g, w)
    tl.store(
        OUT + rm[:, None] * so0 + rko[None, :] * so1,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & ko_mask[None, :],
    )


def mxfp8_grouped_gemm(
    a_bf16: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    m_indices: torch.Tensor,
    tile: int,
    blk_n: int = 128,
) -> torch.Tensor:
    """Grouped ``A @ W^T`` reading mxfp8 W in-register. ``a_bf16`` [Mt, K] bf16 (tile-padded,
    contiguous-grouped); ``wq`` [E, N, K] e4m3; ``ws`` [E, N, K/32] e8m0; ``m_indices`` [Mt/tile]
    int32 expert-id per tile -> [Mt, N] bf16. N need not be ``blk_n``-aligned (masked); K must be a
    whole number of 32-wide MX blocks."""
    Mt, K = a_bf16.shape
    E, N, _ = wq.shape
    assert K % _MX == 0, f"mxfp8_grouped_gemm expects K % 32 == 0, got K={K}"
    ws_u8 = ws.view(torch.uint8)
    out = torch.empty(Mt, N, device=a_bf16.device, dtype=torch.bfloat16)
    grid = (Mt // tile, triton.cdiv(N, blk_n))
    _mxfp8_grouped_fwd_kernel[grid](
        a_bf16,
        wq,
        ws_u8,
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
        ws_u8.stride(0),
        ws_u8.stride(1),
        ws_u8.stride(2),
        out.stride(0),
        out.stride(1),
        BLK_M=tile,
        BLK_N=blk_n,
        BLK_K=_MX,
        num_warps=4,
        num_stages=2,
    )
    return out


def mxfp8_grouped_dx(
    g: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    m_indices: torch.Tensor,
    tile: int,
    blk_n: int = 128,
) -> torch.Tensor:
    """dX = ``g @ W`` (contract N) reading mxfp8 W in-register — the backward companion of
    :func:`mxfp8_grouped_gemm`. ``g`` [Mt, N] bf16; ``wq`` [E, N, Kout] e4m3; ``ws`` [E, N, Kout/32]
    e8m0 -> [Mt, Kout] bf16. Kout must be a whole number of 32-wide MX blocks."""
    Mt, N = g.shape
    E, _, KO = wq.shape
    assert KO % _MX == 0, f"mxfp8_grouped_dx expects Kout % 32 == 0, got Kout={KO}"
    ws_u8 = ws.view(torch.uint8)
    out = torch.empty(Mt, KO, device=g.device, dtype=torch.bfloat16)
    grid = (Mt // tile, triton.cdiv(KO, _MX))
    _mxfp8_grouped_dx_kernel[grid](
        g,
        wq,
        ws_u8,
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
        ws_u8.stride(0),
        ws_u8.stride(1),
        ws_u8.stride(2),
        out.stride(0),
        out.stride(1),
        BLK_M=tile,
        BLK_N=blk_n,
        BLK_KO=_MX,
        num_warps=4,
        num_stages=2,
    )
    return out
