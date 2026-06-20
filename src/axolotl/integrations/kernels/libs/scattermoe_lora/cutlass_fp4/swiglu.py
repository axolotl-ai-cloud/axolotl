"""Fused clamped-SwiGLU forward + backward Triton kernels (memory-lean, no [Mt,I] intermediates).
fwd: gu[Mt,2I] -> h[Mt,I] = silu(clamp(g,max=L)) * clamp(u,-L,L).
bwd: (gu[Mt,2I], dh[Mt,I]) -> dgu[Mt,2I], in ONE pass (the ~8 intermediate tensors the eager
version materialized become register-local)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(GU, H, I, L, sg0, sg1, sh0, sh1, BLK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = c < I
    g = tl.load(GU + r * sg0 + c * sg1, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * sg0 + (I + c) * sg1, mask=m, other=0.0).to(tl.float32)
    gc = tl.minimum(g, L)
    uc = tl.minimum(tl.maximum(u, -L), L)
    h = (gc * tl.sigmoid(gc)) * uc
    tl.store(H + r * sh0 + c * sh1, h.to(H.dtype.element_ty), mask=m)


@triton.jit
def _bwd(GU, DH, DGU, I, L, sg0, sg1, sd0, sd1, so0, so1, BLK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = c < I
    g = tl.load(GU + r * sg0 + c * sg1, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * sg0 + (I + c) * sg1, mask=m, other=0.0).to(tl.float32)
    dh = tl.load(DH + r * sd0 + c * sd1, mask=m, other=0.0).to(tl.float32)
    gc = tl.minimum(g, L)
    sg = tl.sigmoid(gc)
    silu = gc * sg
    dsilu = sg * (1.0 + gc * (1.0 - sg))
    uc = tl.minimum(tl.maximum(u, -L), L)
    dg = dh * uc * dsilu * (g <= L)
    du = dh * silu * ((u >= -L) & (u <= L))
    tl.store(DGU + r * so0 + c * so1, dg.to(DGU.dtype.element_ty), mask=m)
    tl.store(DGU + r * so0 + (I + c) * so1, du.to(DGU.dtype.element_ty), mask=m)


def swiglu_fwd(gu, limit, bpm=512):
    Mt, twoI = gu.shape
    I = twoI // 2
    h = torch.empty(Mt, I, device=gu.device, dtype=gu.dtype)
    _fwd[(Mt, triton.cdiv(I, bpm))](gu, h, I, float(limit), gu.stride(0), gu.stride(1),
                                    h.stride(0), h.stride(1), BLK=bpm)
    return h


def swiglu_bwd(gu, dh, limit, bpm=512):
    Mt, twoI = gu.shape
    I = twoI // 2
    dgu = torch.empty(Mt, twoI, device=gu.device, dtype=gu.dtype)
    _bwd[(Mt, triton.cdiv(I, bpm))](gu, dh, dgu, I, float(limit), gu.stride(0), gu.stride(1),
                                    dh.stride(0), dh.stride(1), dgu.stride(0), dgu.stride(1), BLK=bpm)
    return dgu
