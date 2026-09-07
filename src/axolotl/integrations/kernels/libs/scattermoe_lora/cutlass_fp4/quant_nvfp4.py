"""Fast Triton nvfp4 quantizer: bf16 [M,K] -> qdata u8 [M,K/2] (2 fp4/byte, low-first) +
e4m3 block scale [M,K/16]. Same qdata/scale layout as torchao (drop-in for the grouped FP4 GEMM),
but an approximate fast encode: NOT byte-identical to torchao's to_nvfp4 (arithmetic-encode cascade
+ 1e-30 scale floor here vs torchao's RNE + 2**-6 e4m3 floor)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk(X, Q, S, sx0, sx1, sq0, sq1, ss0, ss1, BPM: tl.constexpr):
    m = tl.program_id(0)
    b0 = tl.program_id(1) * BPM
    blk = tl.arange(0, BPM)
    e = tl.arange(0, 8)  # 8 bytes per 16-wide block
    # even (lo) and odd (hi) fp4 positions interleaved within each byte
    k_lo = (b0 + blk)[:, None] * 16 + 2 * e[None, :]
    k_hi = k_lo + 1
    lo = tl.load(X + m * sx0 + k_lo * sx1).to(tl.float32)
    hi = tl.load(X + m * sx0 + k_hi * sx1).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi), axis=1))
    sc1d = tl.clamp(amax / 6.0, 1e-30, 1e30).to(tl.float8e4nv).to(tl.float32)
    sc1d = tl.where(sc1d > 0, sc1d, 1.0)
    sc = sc1d[:, None]
    alo = tl.abs(lo) / sc
    clo = (
        tl.where(alo > 0.25, 1, 0)
        + tl.where(alo > 0.75, 1, 0)
        + tl.where(alo > 1.25, 1, 0)
        + tl.where(alo > 1.75, 1, 0)
        + tl.where(alo > 2.5, 1, 0)
        + tl.where(alo > 3.5, 1, 0)
        + tl.where(alo > 5.0, 1, 0)
    ).to(tl.int32) + tl.where(lo < 0, 8, 0).to(tl.int32)
    ahi = tl.abs(hi) / sc
    chi = (
        tl.where(ahi > 0.25, 1, 0)
        + tl.where(ahi > 0.75, 1, 0)
        + tl.where(ahi > 1.25, 1, 0)
        + tl.where(ahi > 1.75, 1, 0)
        + tl.where(ahi > 2.5, 1, 0)
        + tl.where(ahi > 3.5, 1, 0)
        + tl.where(ahi > 5.0, 1, 0)
    ).to(tl.int32) + tl.where(hi < 0, 8, 0).to(tl.int32)
    byte = (clo + chi * 16).to(tl.uint8)
    qpos = (b0 + blk)[:, None] * 8 + e[None, :]
    tl.store(Q + m * sq0 + qpos * sq1, byte)
    tl.store(S + m * ss0 + (b0 + blk) * ss1, sc1d.to(tl.float8e4nv))


def nvfp4_quant(x, bpm=8):
    M, K = x.shape
    nblk = K // 16
    Q = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    S = torch.empty(M, nblk, device=x.device, dtype=torch.float8_e4m3fn)
    _qk[(M, triton.cdiv(nblk, bpm))](
        x,
        Q,
        S,
        x.stride(0),
        x.stride(1),
        Q.stride(0),
        Q.stride(1),
        S.stride(0),
        S.stride(1),
        BPM=bpm,
    )
    return Q, S
