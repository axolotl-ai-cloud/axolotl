"""Fused gated-softmax pooling for DeepSeek-V4 compressors (fwd + bwd).

Reference (CSA/HCA compressor §2.3.1/§2.3.2 and the indexer):
    compressed = (kv * gate.softmax(dim=-2, dtype=fp32)).sum(dim=-2)
i.e. per (window, channel) a softmax over the W window tokens, then a weighted sum.
Eager runs softmax (fp32) + multiply + reduce as separate passes over [.., W, Dh];
this fuses them into one kernel, fp32 throughout.

Per output element:  out[m,d] = sum_w p[m,w,d] * kv[m,w,d],  p = softmax_w(gate)
Backward:            d_kv = p * d_out;   d_gate = p * d_out * (kv - out)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BD": bd}, num_warps=w)
        for bd in (32, 64, 128)
        for w in (2, 4, 8)
    ],
    key=["M", "D"],
)
@triton.jit
def _pool_fwd_kernel(KV, GATE, OUT, M, D, W: tl.constexpr, BD: tl.constexpr):
    m = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    dmask = d < D
    wkv = tl.arange(0, W)
    off = m * (W * D) + wkv[:, None] * D + d[None, :]
    cmask = dmask[None, :]
    g = tl.load(GATE + off, mask=cmask, other=float("-inf")).to(tl.float32)
    kv = tl.load(KV + off, mask=cmask, other=0.0).to(tl.float32)
    p = tl.exp(g - tl.max(g, axis=0, keep_dims=True))
    p = p / tl.sum(p, axis=0, keep_dims=True)
    out = tl.sum(p * kv, axis=0)
    tl.store(OUT + m * D + d, out, mask=dmask)


@triton.autotune(
    configs=[
        triton.Config({"BD": bd}, num_warps=w)
        for bd in (32, 64, 128)
        for w in (2, 4, 8)
    ],
    key=["M", "D"],
)
@triton.jit
def _pool_bwd_kernel(
    KV, GATE, DOUT, DKV, DGATE, M, D, W: tl.constexpr, BD: tl.constexpr
):
    m = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    dmask = d < D
    wkv = tl.arange(0, W)
    off = m * (W * D) + wkv[:, None] * D + d[None, :]
    cmask = dmask[None, :]
    g = tl.load(GATE + off, mask=cmask, other=float("-inf")).to(tl.float32)
    kv = tl.load(KV + off, mask=cmask, other=0.0).to(tl.float32)
    p = tl.exp(g - tl.max(g, axis=0, keep_dims=True))
    p = p / tl.sum(p, axis=0, keep_dims=True)
    out = tl.sum(p * kv, axis=0, keep_dims=True)
    dout = tl.load(DOUT + m * D + d, mask=dmask, other=0.0).to(tl.float32)[None, :]
    dkv = p * dout
    dgate = p * dout * (kv - out)
    tl.store(DKV + off, dkv, mask=cmask)
    tl.store(DGATE + off, dgate, mask=cmask)


class _GatedPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, gate):
        M, W, D = kv.shape
        kv = kv.contiguous()
        gate = gate.contiguous()
        out = torch.empty(M, D, device=kv.device, dtype=torch.float32)
        grid = lambda m: (M, triton.cdiv(D, m["BD"]))
        _pool_fwd_kernel[grid](kv, gate, out, M, D, W=W)
        ctx.save_for_backward(kv, gate)
        ctx.dims = (M, W, D)
        return out

    @staticmethod
    def backward(ctx, dout):
        kv, gate = ctx.saved_tensors
        M, W, D = ctx.dims
        dkv = torch.empty_like(kv, dtype=torch.float32)
        dgate = torch.empty_like(gate, dtype=torch.float32)
        grid = lambda m: (M, triton.cdiv(D, m["BD"]))
        _pool_bwd_kernel[grid](kv, gate, dout.contiguous(), dkv, dgate, M, D, W=W)
        return dkv.to(kv.dtype), dgate.to(gate.dtype)


def gated_softmax_pool(kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """kv, gate: [..., W, D]. Returns [..., D] = sum_w softmax_w(gate) * kv (fp32).
    Drop-in for ``(kv * gate.softmax(dim=-2, dtype=fp32)).sum(dim=-2)``. W must be a power
    of 2 (compress_rate / 2*compress_rate are: HCA 128, CSA-overlap 8)."""
    if gate.dtype != kv.dtype:
        gate = gate.to(kv.dtype)
    *lead, W, D = kv.shape
    out = _GatedPool.apply(kv.reshape(-1, W, D), gate.reshape(-1, W, D))
    return out.reshape(*lead, D)
