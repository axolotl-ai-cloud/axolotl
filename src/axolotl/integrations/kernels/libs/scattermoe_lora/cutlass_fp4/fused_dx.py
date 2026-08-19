"""Fused in-kernel-dequant grouped dX Triton kernel (#23): dx[Mt,K] = A[Mt,N] @ Wdeq[expert(m),N,K]
with bf16 grad A + NVFP4 weight decoded IN-KERNEL (no bf16 weight materialization, vectorized, no
Python loop). Contiguous-grouped: each 128-row M-tile belongs to one expert via m_indices.
This is the memory-optimal backward dX (vs dequant+cuBLAS which materializes bf16 W)."""

import torch
import triton
import triton.language as tl


def _codebook(dev):
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=dev,
        dtype=torch.float32,
    )


@triton.jit
def _fused_dx(
    A,
    Wq,
    Ws,
    MI,
    PT,
    OUT,
    CB,
    N,
    K,
    sa0,
    sa1,
    sq0,
    sq1,
    sq2,
    ss0,
    ss1,
    ss2,
    so0,
    so1,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)  # contiguous-grouped: each 128-row M-tile is one expert
    pid_k = tl.program_id(1)
    e = tl.load(MI + pid_m)
    pt_e = tl.load(PT + e).to(tl.float32)
    m = pid_m * 128 + tl.arange(0, 128)
    k = pid_k * BK + tl.arange(0, BK)
    km = k < K
    acc = tl.zeros((128, BK), tl.float32)
    for n0 in range(0, N, BN):
        n = n0 + tl.arange(0, BN)
        a = tl.load(A + m[:, None] * sa0 + n[None, :] * sa1)
        wq = tl.load(
            Wq + e * sq0 + n[:, None] * sq1 + (k[None, :] // 2) * sq2,
            mask=km[None, :],
            other=0,
        ).to(tl.int32)
        nib = tl.where(k[None, :] % 2 == 1, (wq >> 4) & 0xF, wq & 0xF)
        ws = tl.load(
            Ws + e * ss0 + n[:, None] * ss1 + (k[None, :] // 16) * ss2,
            mask=km[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(CB + nib) * ws * pt_e
        acc += tl.dot(a, w.to(tl.bfloat16))
    tl.store(
        OUT + m[:, None] * so0 + k[None, :] * so1, acc.to(tl.bfloat16), mask=km[None, :]
    )


def fused_dx(A, Wq, Ws, m_indices, pt, K, BN=64, BK=64):
    """A[Mt,N] bf16, Wq[E,N,K/2] u8, Ws[E,N,K/16] e4m3, pt[E] -> dx[Mt,K] bf16."""
    Mt, N = A.shape
    # Mt//128 grid covers every row only if 128-padded; otherwise trailing rows stay uninitialized.
    assert Mt % 128 == 0, (
        f"fused_dx expects 128-padded Mt (contiguous-grouped), got Mt={Mt}"
    )
    out = torch.empty(Mt, K, device=A.device, dtype=torch.bfloat16)
    grid = (Mt // 128, triton.cdiv(K, BK))
    _fused_dx[grid](
        A,
        Wq,
        Ws,
        m_indices,
        pt.contiguous(),
        out,
        _codebook(A.device),
        N,
        K,
        A.stride(0),
        A.stride(1),
        Wq.stride(0),
        Wq.stride(1),
        Wq.stride(2),
        Ws.stride(0),
        Ws.stride(1),
        Ws.stride(2),
        out.stride(0),
        out.stride(1),
        BN=BN,
        BK=BK,
    )
    return out
