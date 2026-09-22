"""Fused partial RoPE and MLA Q/K/V assembly for GLM-4.7-Flash."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _prepare_forward(
    Q,
    KV,
    KR,
    COS,
    SIN,
    OQ,
    OK,
    OV,
    KR_STRIDE: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // H
    d = tl.arange(0, BLOCK)
    dtype = Q.dtype.element_ty
    r = d - P
    half = R // 2
    pair = r % half
    a = 2 * pair if INTERLEAVE else pair
    b = a + 1 if INTERLEAVE else a + half
    cs = pair if INTERLEAVE else r
    mask = (r >= 0) & (r < R)
    c = tl.load(COS + token * R + cs, mask, other=0).to(tl.float32)
    s = tl.load(SIN + token * R + cs, mask, other=0).to(tl.float32)
    qa = tl.load(Q + row * (P + R) + P + a, mask, other=0).to(tl.float32)
    qb = tl.load(Q + row * (P + R) + P + b, mask, other=0).to(tl.float32)
    ka = tl.load(KR + token * KR_STRIDE + a, mask, other=0).to(tl.float32)
    kb = tl.load(KR + token * KR_STRIDE + b, mask, other=0).to(tl.float32)
    first = r < half
    qr = (tl.where(first, qa, qb) * c).to(dtype).to(tl.float32) + (
        tl.where(first, -qb, qa) * s
    ).to(dtype).to(tl.float32)
    kr = (tl.where(first, ka, kb) * c).to(dtype).to(tl.float32) + (
        tl.where(first, -kb, ka) * s
    ).to(dtype).to(tl.float32)
    qp = tl.load(Q + row * (P + R) + d, d < P, other=0)
    kp = tl.load(KV + row * (P + V) + d, d < P, other=0)
    v = tl.load(KV + row * (P + V) + P + d, d < V, other=0)
    tl.store(OQ + row * (P + R) + d, tl.where(d < P, qp, qr), d < P + R)
    tl.store(OK + row * (P + R) + d, tl.where(d < P, kp, kr), d < P + R)
    tl.store(OV + row * V + d, v, d < V)


@triton.jit
def _prepare_backward(
    DQ,
    DK,
    DV,
    COS,
    SIN,
    IQ,
    IKV,
    H: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // H
    d = tl.arange(0, BLOCK)
    dtype = DQ.dtype.element_ty
    r = d - P
    half = R // 2
    pair = r // 2 if INTERLEAVE else r % half
    second = r % 2 != 0 if INTERLEAVE else r >= half
    mask = (r >= 0) & (r < R)
    cs1 = pair if INTERLEAVE else pair + half
    c0 = tl.load(COS + token * R + pair, mask, other=0).to(tl.float32)
    c1 = tl.load(COS + token * R + cs1, mask, other=0).to(tl.float32)
    s0 = tl.load(SIN + token * R + pair, mask, other=0).to(tl.float32)
    s1 = tl.load(SIN + token * R + cs1, mask, other=0).to(tl.float32)
    g0 = tl.load(DQ + row * (P + R) + P + pair, mask, other=0).to(tl.float32)
    g1 = tl.load(DQ + row * (P + R) + P + pair + half, mask, other=0).to(tl.float32)
    rot = (tl.where(second, g1 * c1, g0 * c0)).to(dtype).to(tl.float32) + (
        tl.where(second, -g0 * s0, g1 * s1)
    ).to(dtype).to(tl.float32)
    qp = tl.load(DQ + row * (P + R) + d, d < P, other=0)
    kp = tl.load(DK + row * (P + R) + d, d < P, other=0)
    v = tl.load(DV + row * V + d - P, (d >= P) & (d < P + V), other=0)
    tl.store(IQ + row * (P + R) + d, tl.where(d < P, qp, rot), d < P + R)
    tl.store(IKV + row * (P + V) + d, tl.where(d < P, kp, v), d < P + V)


@triton.jit
def _shared_key_backward(
    DK,
    COS,
    SIN,
    IKR,
    H: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.arange(0, HEAD_BLOCK)
    r = tl.arange(0, ROT_BLOCK)
    half = R // 2
    pair = r // 2 if INTERLEAVE else r % half
    second = r % 2 != 0 if INTERLEAVE else r >= half
    offsets = (token * H + heads[:, None]) * (P + R) + P + pair[None, :]
    mask = (heads[:, None] < H) & (r[None, :] < R)
    dtype = DK.dtype.element_ty
    # Sum the expanded heads before rotating, matching the shared-key autograd path.
    g0 = (
        tl.sum(tl.load(DK + offsets, mask, other=0).to(tl.float32), 0)
        .to(dtype)
        .to(tl.float32)
    )
    g1 = (
        tl.sum(tl.load(DK + offsets + half, mask, other=0).to(tl.float32), 0)
        .to(dtype)
        .to(tl.float32)
    )
    cs1 = pair if INTERLEAVE else pair + half
    c0 = tl.load(COS + token * R + pair, r < R, other=0).to(tl.float32)
    c1 = tl.load(COS + token * R + cs1, r < R, other=0).to(tl.float32)
    s0 = tl.load(SIN + token * R + pair, r < R, other=0).to(tl.float32)
    s1 = tl.load(SIN + token * R + cs1, r < R, other=0).to(tl.float32)
    grad = tl.where(second, g1 * c1, g0 * c0).to(dtype).to(tl.float32) + tl.where(
        second, -g0 * s0, g1 * s1
    ).to(dtype).to(tl.float32)
    tl.store(IKR + token * R + r, grad, r < R)


@triton_op("axolotl::glm4_mla_prepare", mutates_args=())
def _prepare(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope: int,
    interleave: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, h, d = q.shape
    r = k_rot.shape[-1]
    v = kv.shape[-1] - nope
    oq = torch.empty((n, h, d), dtype=q.dtype, device=q.device)
    ok = torch.empty_like(oq)
    ov = torch.empty((n, h, v), dtype=q.dtype, device=q.device)
    wrap_triton(_prepare_forward)[(n * h,)](
        q,
        kv,
        k_rot,
        cos,
        sin,
        oq,
        ok,
        ov,
        k_rot.stride(0),
        h,
        nope,
        r,
        v,
        interleave,
        triton.next_power_of_2(max(d, v)),
        enable_fp_fusion=False,
    )
    return oq, ok, ov


@triton_op("axolotl::glm4_mla_prepare_backward", mutates_args=())
def _prepare_grad(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope: int,
    interleave: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, h, d = dq.shape
    r = d - nope
    v = dv.shape[-1]
    iq = torch.empty_like(dq)
    ikv = torch.empty((n, h, nope + v), dtype=dq.dtype, device=dq.device)
    ikr = torch.empty((n, r), dtype=dq.dtype, device=dq.device)
    wrap_triton(_prepare_backward)[(n * h,)](
        dq,
        dk,
        dv,
        cos,
        sin,
        iq,
        ikv,
        h,
        nope,
        r,
        v,
        interleave,
        triton.next_power_of_2(max(d, nope + v)),
        enable_fp_fusion=False,
    )
    wrap_triton(_shared_key_backward)[(n,)](
        dk,
        cos,
        sin,
        ikr,
        h,
        nope,
        r,
        interleave,
        triton.next_power_of_2(h),
        triton.next_power_of_2(r),
        enable_fp_fusion=False,
    )
    return iq, ikv, ikr


def _setup_context(ctx, inputs, output):
    q, kv, _k_rot, cos, sin, nope, interleave = inputs
    ctx.save_for_backward(cos, sin)
    ctx.q_shape = q.shape
    ctx.v_shape = (*q.shape[:-1], kv.shape[-1] - nope)
    ctx.dtype = q.dtype
    ctx.nope = nope
    ctx.interleave = interleave


def _backward(ctx, dq, dk, dv):
    cos, sin = ctx.saved_tensors
    grads = []
    for grad, shape in ((dq, ctx.q_shape), (dk, ctx.q_shape), (dv, ctx.v_shape)):
        grads.append(
            torch.zeros(shape, dtype=ctx.dtype, device=cos.device)
            if grad is None
            else grad.contiguous()
        )
    iq, ikv, ikr = _prepare_grad(*grads, cos, sin, ctx.nope, ctx.interleave)
    return iq, ikv, ikr, None, None, None, None


_prepare.register_autograd(_backward, setup_context=_setup_context)


def fused_mla_prepare(q, kv, k_rot, cos, sin, nope, interleave):
    """Assemble contiguous BSHD Q/K/V from MLA projections and a shared rotary key."""
    b, s, h, d = q.shape
    r = k_rot.shape[-1]
    if nope < 0 or r % 2 or r <= 0 or nope + r != d:
        raise ValueError(
            "MLA requires an even rotary dimension and Q = nonrotary + rotary"
        )
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("Fused MLA does not support trainable rotary embeddings")
    if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError(
            "Fused MLA requires CUDA float16, bfloat16, or float32 tensors"
        )
    if kv.ndim != 4 or kv.shape[:3] != (b, s, h) or k_rot.shape != (b, s, r):
        raise ValueError("Incompatible MLA projection shapes")
    if cos.shape != sin.shape or cos.shape not in ((b, s, r), (1, s, r)):
        raise ValueError(
            "Rotary embeddings must have shape (batch or 1, sequence, rotary)"
        )
    if any(x.dtype != q.dtype or x.device != q.device for x in (kv, k_rot, cos, sin)):
        raise ValueError(
            "MLA projections and rotary embeddings must share dtype and device"
        )
    if kv.shape[-1] <= nope:
        raise ValueError("MLA value dimension must be positive")
    if k_rot.stride(-1) != 1:
        k_rot = k_rot.contiguous()
    cos = cos.expand(b, s, r).reshape(b * s, r).contiguous()
    sin = sin.expand(b, s, r).reshape(b * s, r).contiguous()
    outputs = _prepare(
        q.reshape(b * s, h, d).contiguous(),
        kv.reshape(b * s, h, -1).contiguous(),
        k_rot.reshape(b * s, r),
        cos,
        sin,
        nope,
        interleave,
    )
    return tuple(x.view(b, s, h, -1) for x in outputs)
