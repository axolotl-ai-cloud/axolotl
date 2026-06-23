"""Fused gated-activation forward + backward Triton kernels (memory-lean, no [Mt,I] intermediates).

Two activation variants:
  silu  (default, DSV4): fwd = silu(clamp(g,max=L)) * clamp(u,-L,L)
  gelu_tanh (Gemma4):    fwd = gelu_pytorch_tanh(g) * u  (no clamp)

bwd: (gu[Mt,2I], dh[Mt,I]) -> dgu[Mt,2I] in ONE pass (~8 intermediates become register-local).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


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


# gelu_tanh(x) = 0.5 * x * (1 + tanh(k * (x + 0.044715 * x^3))), k = sqrt(2/pi)
# gelu_tanh'(x) = 0.5*(1+t) + 0.5*x*(1-t^2)*k*(1 + 3*0.044715*x^2)


@triton.jit
def _fwd_gelu(GU, H, I, sg0, sg1, sh0, sh1, BLK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = c < I
    g = tl.load(GU + r * sg0 + c * sg1, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * sg0 + (I + c) * sg1, mask=m, other=0.0).to(tl.float32)
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (g + 0.044715 * g * g * g)
    t = libdevice.tanh(inner)
    gelu_g = 0.5 * g * (1.0 + t)
    h = gelu_g * u
    tl.store(H + r * sh0 + c * sh1, h.to(H.dtype.element_ty), mask=m)


@triton.jit
def _bwd_gelu(GU, DH, DGU, I, sg0, sg1, sd0, sd1, so0, so1, BLK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = c < I
    g = tl.load(GU + r * sg0 + c * sg1, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * sg0 + (I + c) * sg1, mask=m, other=0.0).to(tl.float32)
    dh = tl.load(DH + r * sd0 + c * sd1, mask=m, other=0.0).to(tl.float32)
    k = 0.7978845608028654
    g2 = g * g
    inner = k * (g + 0.044715 * g2 * g)
    t = libdevice.tanh(inner)
    gelu_g = 0.5 * g * (1.0 + t)
    # gelu'(g) = 0.5*(1+t) + 0.5*g*(1-t^2)*k*(1 + 3*0.044715*g^2)
    dgelu = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * g2)
    dg = dh * u * dgelu
    du = dh * gelu_g
    tl.store(DGU + r * so0 + c * so1, dg.to(DGU.dtype.element_ty), mask=m)
    tl.store(DGU + r * so0 + (I + c) * so1, du.to(DGU.dtype.element_ty), mask=m)


def swiglu_fwd(gu, limit, act_type="silu", bpm=512):
    """Gated-activation forward: gu[Mt,2I] -> h[Mt,I].
    act_type='silu': clamped SwiGLU (DSV4). act_type='gelu_tanh': GeGLU (Gemma4, limit ignored)."""
    Mt, twoI = gu.shape
    I = twoI // 2
    h = torch.empty(Mt, I, device=gu.device, dtype=gu.dtype)
    if act_type == "gelu_tanh":
        _fwd_gelu[(Mt, triton.cdiv(I, bpm))](
            gu, h, I, gu.stride(0), gu.stride(1), h.stride(0), h.stride(1), BLK=bpm
        )
    else:
        _fwd[(Mt, triton.cdiv(I, bpm))](
            gu,
            h,
            I,
            float(limit),
            gu.stride(0),
            gu.stride(1),
            h.stride(0),
            h.stride(1),
            BLK=bpm,
        )
    return h


def swiglu_bwd(gu, dh, limit, act_type="silu", bpm=512):
    """Gated-activation backward: (gu[Mt,2I], dh[Mt,I]) -> dgu[Mt,2I].
    act_type='silu': clamped SwiGLU (DSV4). act_type='gelu_tanh': GeGLU (Gemma4, limit ignored)."""
    Mt, twoI = gu.shape
    I = twoI // 2
    dgu = torch.empty(Mt, twoI, device=gu.device, dtype=gu.dtype)
    if act_type == "gelu_tanh":
        _bwd_gelu[(Mt, triton.cdiv(I, bpm))](
            gu,
            dh,
            dgu,
            I,
            gu.stride(0),
            gu.stride(1),
            dh.stride(0),
            dh.stride(1),
            dgu.stride(0),
            dgu.stride(1),
            BLK=bpm,
        )
    else:
        _bwd[(Mt, triton.cdiv(I, bpm))](
            gu,
            dh,
            dgu,
            I,
            float(limit),
            gu.stride(0),
            gu.stride(1),
            dh.stride(0),
            dh.stride(1),
            dgu.stride(0),
            dgu.stride(1),
            BLK=bpm,
        )
    return dgu
