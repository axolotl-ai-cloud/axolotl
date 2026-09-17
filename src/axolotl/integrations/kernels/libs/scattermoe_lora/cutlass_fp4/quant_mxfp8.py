"""Fast Triton mxfp8 quantizer for activations (fp8 mode): bf16 [M,K] -> qdata e4m3 [M,K] +
e8m0 block scale [M,K/32]. Matches torchao MXTensor.to_mx(x, float8_e4m3fn, 32, FLOOR)."""

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _qk(X, Q, S, sx0, sx1, sq0, sq1, ss0, ss1, BPM: tl.constexpr):
    m = tl.program_id(0)
    b0 = tl.program_id(1) * BPM
    blk = tl.arange(0, BPM)
    e = tl.arange(0, 32)
    k = (b0 + blk)[:, None] * 32 + e[None, :]
    x = tl.load(X + m * sx0 + k * sx1).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    # FLOOR e8m0 scale: 2^(floor(log2(amax)) - floor(log2(448))), floor(log2(448))=8
    exp = tl.floor(tl.log2(tl.where(amax > 0, amax, 1.0))) - 8.0
    sc = tl.exp2(exp)
    scb = sc[:, None]
    q = tl.clamp(x / scb, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Q + m * sq0 + k * sq1, q)
    # e8m0 stores the biased exponent byte (bias 127); scale = 2^(E-127)
    Ebiased = (exp + 127.0).to(tl.int32)
    tl.store(S + m * ss0 + (b0 + blk) * ss1, Ebiased.to(tl.uint8))


def mxfp8_quant(x, bpm=4):
    M, K = x.shape
    # K must be a whole number of 32-wide MX blocks, else nblk drops the tail and the kernel OOBs.
    assert K % 32 == 0, f"mxfp8_quant expects K divisible by 32, got K={K}"
    nblk = K // 32
    Q = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    S = torch.empty(
        M, nblk, device=x.device, dtype=torch.uint8
    )  # e8m0 biased-exponent bytes
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
    return Q, S.view(torch.float8_e8m0fnu)
