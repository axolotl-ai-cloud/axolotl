# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Fused LoRA attention projections (QKV / QK) for Blackwell sm_120.

Q/K/V (or Q/K) are fused into a single GEMM by concatenating their base weights,
so X is read once. Per-projection LoRA stays block-diagonal, so each adapter only
touches its own output slice (ranks may differ per projection). The output is
split back into the individual projections.

QKV uses a monolithic autograd Function (`_LoRAQKV`) with a hand-optimized,
in-place backward (computes each adapter grad directly, runs dX through the fused
GEMM written over X's buffer) — same discipline that makes the fused MLP a
training win. QK (rarer, v_proj reused) stays on the composed path.

Only raw projections are computed here; arch-specific bits (q_norm/k_norm, RoPE,
GQA reshape, gating) happen downstream, so this is architecture-agnostic.
"""

import torch

from .autograd import lora_dense
from .forward import lora_dense_forward

# Frozen base weights -> the concatenation is a one-time cost; cache by identity.
_wcat_cache: dict = {}


def _concat_weights(weights: tuple) -> torch.Tensor:
    key = tuple(w.data_ptr() for w in weights)
    cat = _wcat_cache.get(key)
    total = sum(w.shape[0] for w in weights)
    if cat is None or cat.shape[0] != total:
        cat = torch.cat([w.detach() for w in weights], dim=0)
        _wcat_cache[key] = cat
    return cat


def _block_diag(blocks: list) -> torch.Tensor:
    """Block-diagonal of [Di, ri] LoRA-B blocks -> [sum Di, sum ri]."""
    total_c = sum(b.shape[1] for b in blocks)
    rows, c = [], 0
    for b in blocks:
        Di, ri = b.shape
        left = torch.zeros(Di, c, device=b.device, dtype=b.dtype)
        right = torch.zeros(Di, total_c - c - ri, device=b.device, dtype=b.dtype)
        rows.append(torch.cat([left, b, right], dim=1))
        c += ri
    return torch.cat(rows, dim=0)


class _LoRAQKV(torch.autograd.Function):
    """Monolithic fused Q/K/V projection. Forward is one fused GEMM over the
    concatenated weight (X read once, LoRA epilogue in-register -> no [M,Di]
    intermediates), returning Q/K/V as separate views. Backward computes each
    adapter grad directly and accumulates dX in-place over X's buffer via cuBLAS
    (no [M,ΣD] grad concat, no fresh dX) — strictly less memory than the existing
    per-projection path."""

    @staticmethod
    def forward(ctx, X, qW, qA, qB, kW, kA, kB, vW, vA, vB, scaling):
        Dq, Dk, Dv = qW.shape[0], kW.shape[0], vW.shape[0]
        W_qkv = _concat_weights((qW.detach(), kW.detach(), vW.detach()))
        A_qkv = torch.cat([qA, kA, vA], dim=0).detach()
        B_qkv = _block_diag([qB.detach(), kB.detach(), vB.detach()])
        out = lora_dense_forward(X.detach(), W_qkv, A_qkv, B_qkv, scaling)  # [M, ΣD]
        ctx.save_for_backward(X, qA, qB, kA, kB, vA, vB, qW, kW, vW)
        ctx.scaling = scaling
        return out.split((Dq, Dk, Dv), dim=-1)

    @staticmethod
    def backward(ctx, dQ, dK, dV):
        X, qA, qB, kA, kB, vA, vB, qW, kW, vW = ctx.saved_tensors
        s = ctx.scaling
        dQ, dK, dV = dQ.contiguous(), dK.contiguous(), dV.contiguous()

        # per-projection LoRA grads (consume X before dX overwrites its buffer)
        DGBq, DGBk, DGBv = dQ @ qB, dK @ kB, dV @ vB  # [M, ri]
        dqA, dkA, dvA = s * (DGBq.t() @ X), s * (DGBk.t() @ X), s * (DGBv.t() @ X)
        dqB = s * (dQ.t() @ (X @ qA.t()))  # [Dq, rq]
        dkB = s * (dK.t() @ (X @ kA.t()))
        dvB = s * (dV.t() @ (X @ vA.t()))

        dX = None
        if ctx.needs_input_grad[0]:
            # dX = Σ dP@Wp + s·Σ (dP@pB)@pA, accumulated in-place over X's buffer
            # (X is fully consumed above) — avoids a [M,ΣD] concat and a fresh dX.
            dX = X.detach()
            dX.zero_()
            dX.addmm_(dQ, qW)
            dX.addmm_(dK, kW)
            dX.addmm_(dV, vW)
            dX.addmm_(DGBq, qA, alpha=s)
            dX.addmm_(DGBk, kA, alpha=s)
            dX.addmm_(DGBv, vA, alpha=s)
        # grads: X, qW,qA,qB, kW,kA,kB, vW,vA,vB, scaling
        return dX, None, dqA, dqB, None, dkA, dkB, None, dvA, dvB, None


def _cast_adapters(dt, *ts):
    return [t.to(dt) for t in ts]


def lora_qkv(X, qW, qA, qB, kW, kA, kB, vW, vA, vB, scaling):
    """Fused Q/K/V projections -> (Q, K, V). GQA-friendly (K/V can be smaller)."""
    flat = X.reshape(-1, X.shape[-1]) if X.dim() != 2 else X
    qA, qB, kA, kB, vA, vB = _cast_adapters(flat.dtype, qA, qB, kA, kB, vA, vB)
    Q, K, V = _LoRAQKV.apply(flat, qW, qA, qB, kW, kA, kB, vW, vA, vB, scaling)
    if X.dim() != 2:
        Q = Q.reshape(*X.shape[:-1], -1)
        K = K.reshape(*X.shape[:-1], -1)
        V = V.reshape(*X.shape[:-1], -1)
    return Q, K, V


def lora_qk(X, qW, qA, qB, kW, kA, kB, scaling):
    """Fused Q/K projections -> (Q, K) (composed path; rarer v_proj-reused case)."""
    W = _concat_weights((qW, kW))
    A = torch.cat([qA, kA], dim=0)
    B = _block_diag([qB, kB])
    out = lora_dense(X, W, A, B, scaling)
    return out.split((qW.shape[0], kW.shape[0]), dim=-1)
