"""Sparse top-k MLA attention for GLM-5.2 DSA (forward + backward, training-correct).

Not on the live training path: the runtime dispatcher (``dispatch.py``/``patch.py``) uses the
weight-absorption kernel in ``attention_mla_absorb`` instead. This head-per-program baseline is
retained only as an autotune/benchmarking reference (see ``autotune_collector``).

The DSA training path (eager/sdpa) scatters the per-query top-k indices into a dense ``[B,S,T]``
additive mask and runs full attention over all T keys — O(S^2) memory + compute even though only
``index_topk`` (2048) keys per query matter. This kernel **gathers** each query's selected keys and
runs flash online-softmax over just those: O(S·topk).

``sparse_attn`` is a ``torch.autograd.Function`` — fwd + bwd — so it composes with training. It
differentiates wrt q, k, v (the MLA projection activations), so gradients flow to the projection
weights in **full-parameter** training and to the LoRA adapters in **LoRA** training identically;
the kernel is agnostic to how q/k/v were produced. The top-k indices are treated as constants
(``topk`` is non-differentiable, exactly as in the eager model), and the DSA indexer itself is
``@torch.no_grad`` upstream, so no gradient flows through key selection.

Layout: q,k ``[B,H,S,D]`` (D=qk_head_dim=256), v ``[B,H,S,Dv]`` (Dv=v_head_dim=256), ``topk_idx``
``[B,S,T]`` int32 (selected key positions, shared across heads). Gathered indices beyond the query
position are causal-masked so over-selected top-k padding contributes nothing (fwd) and gets no
gradient (bwd). One program = (b, h, query position s); MMA-efficient batching is a later
optimization — this is the correct, gradient-validated baseline.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from axolotl.kernels.op_registry import register_kernel_op


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=w, num_stages=s)
        for bt in (64, 128, 256)
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["TOPK", "D", "DV"],
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    IDX,
    OUT,
    LSE,
    scale,
    S,
    TOPK,
    sq_b,
    sq_h,
    sq_s,
    sq_d,
    sk_b,
    sk_h,
    sk_s,
    sk_d,
    sv_b,
    sv_h,
    sv_s,
    sv_d,
    si_b,
    si_s,
    si_t,
    so_b,
    so_h,
    so_s,
    so_d,
    sl_b,
    sl_h,
    sl_s,
    H,
    D: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)
    q = tl.load(Q + b * sq_b + h * sq_h + s * sq_s + offs_d * sq_d).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((DV,), dtype=tl.float32)

    for t0 in range(0, TOPK, BT):
        offs_t = t0 + tl.arange(0, BT)
        tmask = offs_t < TOPK
        idx = tl.load(
            IDX + b * si_b + s * si_s + offs_t * si_t, mask=tmask, other=0
        ).to(tl.int64)
        valid = tmask & (idx <= s)

        k_ptr = K + b * sk_b + h * sk_h + idx[:, None] * sk_s + offs_d[None, :] * sk_d
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BT, D]
        qk = tl.sum(q[None, :] * k, axis=1) * scale  # [BT]
        qk = tl.where(valid, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v_ptr = V + b * sv_b + h * sv_h + idx[:, None] * sv_s + offs_dv[None, :] * sv_d
        v = tl.load(v_ptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BT, DV]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(
        OUT + b * so_b + h * so_h + s * so_s + offs_dv * so_d,
        acc.to(OUT.dtype.element_ty),
    )
    tl.store(LSE + b * sl_b + h * sl_h + s * sl_s, m_i + tl.log(l_i))


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=w, num_stages=s)
        for bt in (64, 128, 256)
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["TOPK", "D", "DV"],
    # dK/dV are atomic-accumulated; zero them between autotune trials or benchmarking corrupts them.
    reset_to_zero=["DKO", "DVO"],
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    IDX,
    OUT,
    DOUT,
    LSE,
    DQO,
    DKO,
    DVO,
    scale,
    S,
    TOPK,
    sq_b,
    sq_h,
    sq_s,
    sq_d,
    sk_b,
    sk_h,
    sk_s,
    sk_d,
    sv_b,
    sv_h,
    sv_s,
    sv_d,
    si_b,
    si_s,
    si_t,
    so_b,
    so_h,
    so_s,
    so_d,
    sl_b,
    sl_h,
    sl_s,
    H,
    D: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)
    q = tl.load(Q + b * sq_b + h * sq_h + s * sq_s + offs_d * sq_d).to(tl.float32)
    dout = tl.load(DOUT + b * so_b + h * so_h + s * so_s + offs_dv * so_d).to(
        tl.float32
    )
    out = tl.load(OUT + b * so_b + h * so_h + s * so_s + offs_dv * so_d).to(tl.float32)
    lse = tl.load(LSE + b * sl_b + h * sl_h + s * sl_s)
    delta = tl.sum(dout * out, axis=0)  # = Σ_j p_j (dout·V_j)

    dq = tl.zeros((D,), dtype=tl.float32)
    for t0 in range(0, TOPK, BT):
        offs_t = t0 + tl.arange(0, BT)
        tmask = offs_t < TOPK
        idx = tl.load(
            IDX + b * si_b + s * si_s + offs_t * si_t, mask=tmask, other=0
        ).to(tl.int64)
        valid = tmask & (idx <= s)

        k_ptr = K + b * sk_b + h * sk_h + idx[:, None] * sk_s + offs_d[None, :] * sk_d
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BT, D]
        v_ptr = V + b * sv_b + h * sv_h + idx[:, None] * sv_s + offs_dv[None, :] * sv_d
        v = tl.load(v_ptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BT, DV]

        s_j = tl.sum(q[None, :] * k, axis=1) * scale  # [BT]
        p = tl.where(valid, tl.exp(s_j - lse), 0.0)  # [BT]
        dp = tl.sum(dout[None, :] * v, axis=1)  # [BT] = dout·V_j
        ds = p * (dp - delta)  # [BT] grad wrt pre-softmax score

        dq += tl.sum((ds * scale)[:, None] * k, axis=0)  # [D]
        # scatter-add dK_j = scale·ds_j·q, dV_j = p_j·dout  (race across query positions -> atomic)
        dk_j = (ds * scale)[:, None] * q[None, :]  # [BT, D]
        dv_j = p[:, None] * dout[None, :]  # [BT, DV]
        dk_ptr = (
            DKO + b * sk_b + h * sk_h + idx[:, None] * sk_s + offs_d[None, :] * sk_d
        )
        dv_ptr = (
            DVO + b * sv_b + h * sv_h + idx[:, None] * sv_s + offs_dv[None, :] * sv_d
        )
        tl.atomic_add(dk_ptr, tl.where(valid[:, None], dk_j, 0.0))
        tl.atomic_add(dv_ptr, tl.where(valid[:, None], dv_j, 0.0))

    tl.store(DQO + b * sq_b + h * sq_h + s * sq_s + offs_d * sq_d, dq)


@register_kernel_op("glm_dsa_sparse_topk_attn_fwd")
def _sparse_topk_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = q.shape
    DV = v.shape[-1]
    TOPK = idx.shape[-1]
    out = torch.empty(B, H, S, DV, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    grid = (B * H, S)
    _fwd_kernel[grid](
        q,
        k,
        v,
        idx,
        out,
        lse,
        scale,
        S,
        TOPK,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        idx.stride(0),
        idx.stride(1),
        idx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        H,
        D=D,
        DV=DV,
    )
    return out, lse


@_sparse_topk_fwd_op.register_fake
def _(q, k, v, idx, scale):
    B, H, S, _ = q.shape
    return (
        torch.empty(B, H, S, v.shape[-1], device=q.device, dtype=q.dtype),
        torch.empty(B, H, S, device=q.device, dtype=torch.float32),
    )


@register_kernel_op("glm_dsa_sparse_topk_attn_bwd")
def _sparse_topk_bwd_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, D = q.shape
    DV = v.shape[-1]
    TOPK = idx.shape[-1]
    dout = dout.contiguous()
    dq = torch.zeros_like(q, dtype=torch.float32)
    # dK/dV accumulate across query positions selecting the same key -> fp32 atomics.
    dk = torch.zeros(B, H, S, D, device=q.device, dtype=torch.float32)
    dv = torch.zeros(B, H, S, DV, device=q.device, dtype=torch.float32)
    grid = (B * H, S)
    _bwd_kernel[grid](
        q,
        k,
        v,
        idx,
        out,
        dout,
        lse,
        dq,
        dk,
        dv,
        scale,
        S,
        TOPK,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        idx.stride(0),
        idx.stride(1),
        idx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        H,
        D=D,
        DV=DV,
    )
    return dq, dk, dv


@_sparse_topk_bwd_op.register_fake
def _(dout, q, k, v, idx, out, lse, scale):
    B, H, S, D = q.shape
    DV = v.shape[-1]
    return (
        torch.empty_like(q, dtype=torch.float32),
        torch.empty(B, H, S, D, device=q.device, dtype=torch.float32),
        torch.empty(B, H, S, DV, device=q.device, dtype=torch.float32),
    )


class _SparseAttnTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk_idx, scale):
        H = q.shape[1]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        idx = topk_idx.contiguous()
        out, lse = _sparse_topk_fwd_op(q, k, v, idx, float(scale))
        ctx.save_for_backward(q, k, v, idx, out, lse)
        ctx.scale = float(scale)
        ctx.H = H
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, idx, out, lse = ctx.saved_tensors
        dq, dk, dv = _sparse_topk_bwd_op(dout, q, k, v, idx, out, lse, ctx.scale)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def sparse_attn(q, k, v, topk_idx, scale):
    """Differentiable sparse top-k MLA attention. q,k [B,H,S,D], v [B,H,S,Dv], topk_idx [B,S,T]
    int32. Returns out [B,H,S,Dv]; backprops dq/dk/dv (topk_idx is a constant)."""
    return _SparseAttnTopK.apply(q, k, v, topk_idx, scale)


def sparse_attn_fwd(q, k, v, topk_idx, scale):
    """Forward-only (no autograd) — for benchmarking the kernel in isolation."""
    with torch.no_grad():
        return _SparseAttnTopK.apply(
            q.detach(), k.detach(), v.detach(), topk_idx, scale
        )
