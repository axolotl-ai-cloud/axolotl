# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Fused LoRA GLU-MLP (SwiGLU / GeGLU) for Blackwell sm_120.

gate = X@Wg^T + s*(X@Ag^T)@Bg^T
up   = X@Wu^T + s*(X@Au^T)@Bu^T
h    = act(gate) * up            # act = SiLU (swiglu) or exact GELU (geglu)
out  = h@Wd^T + s*(h@Ad^T)@Bd^T

gate and up are fused into a single GEMM by concatenating their weights, so X is
read once and shared. The LoRA correction is kept block-diagonal so the fused
epilogue kernel applies both adapters in-register (no [M,inter] LoRA intermediates).
The activation is a fused strided Triton fwd/bwd over the [M,2I] gate_up tensor;
the down projection reuses the same fused LoRA GEMM. Trainable via a monolithic
autograd Function with a hand-optimized, in-place backward.
"""

import torch
import triton
import triton.language as tl

from .autograd import _wt_contig
from .forward import (
    _MMA_K,
    _round_up,
    _run_lora_gemm,
    lora_dense_forward,
    pad_rank,
)

_ACT_SILU: tl.constexpr = 0
_ACT_GELU: tl.constexpr = 1


@triton.jit
def _act(g, ACT: tl.constexpr):
    """GLU activation in fp32: SiLU or exact (erf) GELU."""
    if ACT == 0:
        return g * tl.sigmoid(g)
    return 0.5 * g * (tl.math.erf(tl.math.rsqrt(2.0) * g) + 1.0)


@triton.jit
def _act_grad(g, ACT: tl.constexpr):
    """d/dg of the GLU activation, in fp32."""
    if ACT == 0:
        sig = tl.sigmoid(g)
        return sig + g * sig * (1.0 - sig)
    cdf = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * g) + 1.0)
    pdf = 0.3989422804014327 * tl.exp(-0.5 * g * g)  # 1/sqrt(2*pi)
    return cdf + g * pdf


@triton.jit
def _gateup_act_kernel(
    gu_ptr, out_ptr, M, inter, stride_gu_m, ACT: tl.constexpr, BLOCK: tl.constexpr
):
    """h[m,i] = act(gate_up[m,i]) * gate_up[m, inter+i] for a [M, 2I] gate_up tensor,
    reading both halves directly (no contiguous copies)."""
    pid = tl.program_id(0)
    row = pid // tl.cdiv(inter, BLOCK)
    col0 = (pid % tl.cdiv(inter, BLOCK)) * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    mask = (row < M) & (cols < inter)
    base = row * stride_gu_m
    g = tl.load(gu_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    h = (_act(g, ACT) * u).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * inter + cols, h, mask=mask)


@triton.jit
def _gateup_act_bwd_kernel(
    gu_ptr, dh_ptr, dgu_ptr, M, inter, stride_m, ACT: tl.constexpr, BLOCK: tl.constexpr
):
    """From dh[M,inter] and gate_up[M,2I], write dgate_up[M,2I]:
    dg = dh * u * act'(g),  du = dh * act(g)."""
    pid = tl.program_id(0)
    nblk = tl.cdiv(inter, BLOCK)
    row = pid // nblk
    cols = (pid % nblk) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < M) & (cols < inter)
    base = row * stride_m
    g = tl.load(gu_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dh_ptr + row * inter + cols, mask=mask, other=0.0).to(tl.float32)
    dg = dy * u * _act_grad(g, ACT)
    du = dy * _act(g, ACT)
    tl.store(dgu_ptr + base + cols, dg.to(dgu_ptr.dtype.element_ty), mask=mask)
    tl.store(dgu_ptr + base + inter + cols, du.to(dgu_ptr.dtype.element_ty), mask=mask)


def _glu_fwd(gate_up: torch.Tensor, inter: int, act: int) -> torch.Tensor:
    """h = act(gate_up[:, :inter]) * gate_up[:, inter:]  -> [M, inter]; reads halves in-place."""
    M = gate_up.shape[0]
    out = torch.empty(M, inter, device=gate_up.device, dtype=gate_up.dtype)
    BLOCK = 1024
    grid = (M * triton.cdiv(inter, BLOCK),)
    _gateup_act_kernel[grid](
        gate_up, out, M, inter, gate_up.stride(0), act, BLOCK=BLOCK
    )
    return out


def _glu_bwd(
    gate_up: torch.Tensor, dh: torch.Tensor, inter: int, act: int
) -> torch.Tensor:
    """dgate_up[M, 2I] from dh[M, inter] and gate_up[M, 2I], written IN-PLACE over
    gate_up (each program reads its g/u before writing dg/du, so this is safe and
    avoids a second [M,2I] buffer). gate_up must not be needed afterwards."""
    M = gate_up.shape[0]
    BLOCK = 1024
    grid = (M * triton.cdiv(inter, BLOCK),)
    _gateup_act_bwd_kernel[grid](
        gate_up, dh.contiguous(), gate_up, M, inter, gate_up.stride(0), act, BLOCK=BLOCK
    )
    return gate_up


# Base weights are frozen in LoRA, so the gate/up concatenation is a one-time
# setup cost (not per-step). Cache it keyed by the source weights' identities.
_wgu_cache: dict = {}


def _concat_gate_up_weight(gate_W: torch.Tensor, up_W: torch.Tensor) -> torch.Tensor:
    key = (gate_W.data_ptr(), up_W.data_ptr())
    W_gu = _wgu_cache.get(key)
    if W_gu is None or W_gu.shape[0] != gate_W.shape[0] + up_W.shape[0]:
        W_gu = torch.cat([gate_W, up_W], dim=0)
        _wgu_cache[key] = W_gu
    return W_gu


def _blockdiag_B(gate_B, up_B, inter, r):
    """[[gate_B, 0], [0, up_B]] -> [2I, 2r] (detached; backward is manual)."""
    z = torch.zeros(inter, r, device=gate_B.device, dtype=gate_B.dtype)
    return torch.cat(
        [torch.cat([gate_B, z], dim=1), torch.cat([z, up_B], dim=1)], dim=0
    )


class _LoRAMLPGLU(torch.autograd.Function):
    """Monolithic fused LoRA GLU-MLP (SwiGLU or GeGLU). Hand-optimized backward:
    saves only X and gate_up, recomputes h, runs dX/dh through the fused GEMM,
    frees intermediates eagerly, computes each adapter grad directly."""

    @staticmethod
    def forward(ctx, X, gW, gA, gB, uW, uA, uB, dW, dA, dB, scaling, act):
        inter, r = gW.shape[0], gA.shape[0]
        W_gu = _concat_gate_up_weight(gW.detach(), uW.detach())
        A_gu = torch.cat([gA, uA], dim=0).detach()
        B_gu = _blockdiag_B(gB.detach(), uB.detach(), inter, r)
        gate_up = lora_dense_forward(X.detach(), W_gu, A_gu, B_gu, scaling)  # [M,2I]
        h = _glu_fwd(gate_up, inter, act)  # [M,inter]
        out = lora_dense_forward(h, dW.detach(), dA.detach(), dB.detach(), scaling)
        ctx.save_for_backward(X, gA, gB, uA, uB, dA, dB, gW, uW, dW, gate_up)
        ctx.scaling, ctx.inter, ctx.r, ctx.act = scaling, inter, r, act
        return out

    @staticmethod
    def backward(ctx, dout):
        X, gA, gB, uA, uB, dA, dB, gW, uW, dW, gate_up = ctx.saved_tensors
        s, inter, r, act = ctx.scaling, ctx.inter, ctx.r, ctx.act
        rp = _round_up(max(r, _MMA_K), _MMA_K)
        rp2 = _round_up(max(2 * r, _MMA_K), _MMA_K)
        dout = dout.contiguous()

        h = _glu_fwd(gate_up, inter, act)  # recompute, don't save
        # ---- down backward ----
        DoB = dout @ dB  # [M, r]
        ddA = s * (DoB.t() @ h)  # [r, inter]
        ddB = s * (dout.t() @ (h @ dA.t()))  # [H, r]
        WdT = _wt_contig(dW)  # [inter, H]
        # h is fully consumed above; reuse its buffer for dh to avoid a [M,inter] alloc.
        dh = _run_lora_gemm(
            dout,
            WdT,
            pad_rank(s * DoB, r, rp),
            pad_rank(dA.t().contiguous(), r, rp),
            out=h.detach(),
        )  # [M, inter], over h
        del DoB, dout

        # ---- activation backward ----
        dgate_up = _glu_bwd(gate_up, dh, inter, act)  # [M, 2I]
        del dh
        dgate, dup = dgate_up[:, :inter], dgate_up[:, inter:]
        # ---- gate/up adapter grads (direct, block-diagonal) ----
        DGBg, DGBu = dgate @ gB, dup @ uB  # [M, r] each
        dgA = s * (DGBg.t() @ X)  # [r, H]
        duA = s * (DGBu.t() @ X)
        dgB = s * (dgate.t() @ (X @ gA.t()))  # [inter, r]
        duB = s * (dup.t() @ (X @ uA.t()))
        # ---- dX through the fused GEMM (one call over the concat) ----
        dX = None
        if ctx.needs_input_grad[0]:
            W_gu = _concat_gate_up_weight(gW, uW)
            WguT = _wt_contig(W_gu)  # [H, 2I]
            DGB = torch.cat([s * DGBg, s * DGBu], dim=1)  # [M, 2r]
            AguT = torch.cat([gA.t(), uA.t()], dim=1)  # [H, 2r]
            # X is fully consumed above, so reuse its buffer for dX (like
            # LoRA_MLP's inplace=True) to avoid a fresh [M, H] allocation.
            dX = _run_lora_gemm(
                dgate_up,
                WguT,
                pad_rank(DGB, 2 * r, rp2),
                pad_rank(AguT.contiguous(), 2 * r, rp2),
                out=X.detach(),
            )  # [M, H], over X
        # grads: X, gW, gA, gB, uW, uA, uB, dW, dA, dB, scaling, act
        return dX, None, dgA, dgB, None, duA, duB, None, ddA, ddB, None, None


def _lora_mlp_glu(
    X, gate_W, gate_A, gate_B, up_W, up_A, up_B, down_W, down_A, down_B, scaling, act
):
    flat = X.reshape(-1, X.shape[-1]) if X.dim() != 2 else X
    # PEFT keeps LoRA adapters in fp32 while the base is bf16; cast them to the
    # input dtype (differentiable, so grads still flow back to the fp32 params).
    dt = flat.dtype
    gate_A, gate_B = gate_A.to(dt), gate_B.to(dt)
    up_A, up_B = up_A.to(dt), up_B.to(dt)
    down_A, down_B = down_A.to(dt), down_B.to(dt)
    out = _LoRAMLPGLU.apply(
        flat,
        gate_W,
        gate_A,
        gate_B,
        up_W,
        up_A,
        up_B,
        down_W,
        down_A,
        down_B,
        scaling,
        act,
    )
    return out.reshape(*X.shape[:-1], down_W.shape[0]) if X.dim() != 2 else out


def lora_mlp_swiglu(
    X, gate_W, gate_A, gate_B, up_W, up_A, up_B, down_W, down_A, down_B, scaling
):
    """Trainable fused LoRA SwiGLU MLP (base frozen; trains the A/B adapters).
    X may be [*, H]; gate/up_W=[inter,H], down_W=[H,inter], *_A=[r,*], *_B=[*,r]."""
    return _lora_mlp_glu(
        X, gate_W, gate_A, gate_B, up_W, up_A, up_B, down_W, down_A, down_B, scaling, 0
    )  # SiLU


def lora_mlp_geglu(
    X, gate_W, gate_A, gate_B, up_W, up_A, up_B, down_W, down_A, down_B, scaling
):
    """Trainable fused LoRA GeGLU MLP (exact/erf GELU). Same fusion as swiglu."""
    return _lora_mlp_glu(
        X, gate_W, gate_A, gate_B, up_W, up_A, up_B, down_W, down_A, down_B, scaling, 1
    )  # GELU


def lora_mlp_swiglu_forward(*args, **kwargs):
    """Inference-only alias (no_grad) of the fused MLP."""
    with torch.no_grad():
        return lora_mlp_swiglu(*args, **kwargs)
