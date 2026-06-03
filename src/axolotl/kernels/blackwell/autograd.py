# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Trainable autograd.Function wrappers for the fused sm_120 LoRA kernels.

The base weight W is frozen in LoRA, so backward produces only dX (to propagate)
and dA/dB (the adapter grads). dX reuses the fused GEMM with remapped operands
(dX = dY@W + s*(dY@B)@A); dA/dB are small token-dim reductions on cuBLAS.
"""

import torch

from .forward import _MMA_K, _round_up, _run_lora_gemm, lora_dense_forward, pad_rank

# W is frozen, so its transpose (needed for the dX GEMM) is a one-time cost.
_wt_cache: dict = {}


def _wt_contig(W: torch.Tensor) -> torch.Tensor:
    key = W.data_ptr()
    wt = _wt_cache.get(key)
    if wt is None or tuple(wt.shape) != (W.shape[1], W.shape[0]):
        wt = W.t().contiguous()
        _wt_cache[key] = wt
    return wt


class LoRADense(torch.autograd.Function):
    """Y = X @ W^T + scaling * (X @ A^T) @ B^T, with X/A/B differentiable (W frozen)."""

    @staticmethod
    def forward(ctx, X, W, A, B, scaling):
        ctx.save_for_backward(X, W, A, B)
        ctx.scaling = scaling
        # kernel consumes tensors via dlpack, which rejects grad-requiring tensors;
        # backward is computed manually so detaching here is correct.
        return lora_dense_forward(
            X.detach(), W.detach(), A.detach(), B.detach(), scaling
        )

    @staticmethod
    def backward(ctx, dY):
        X, W, A, B = ctx.saved_tensors
        s = ctx.scaling
        r = A.shape[0]
        r_pad = _round_up(max(r, _MMA_K), _MMA_K)
        dY = dY.contiguous()

        DYB = dY @ B  # [M, r]
        # dX = dY@W + s*(DYB)@A  — fused: base dY@W^T-mapped + epilogue s*DYB@A
        dX = None
        if ctx.needs_input_grad[0]:
            WT = _wt_contig(W)  # [K, N]  (dY@W via X@W^T mapping)
            XA_dx = pad_rank(s * DYB, r, r_pad)  # [M, r_pad]
            B_dx = pad_rank(A.t().contiguous(), r, r_pad)  # [K, r_pad]
            dX = _run_lora_gemm(dY, WT, XA_dx, B_dx)  # [M, K]

        dA = dB = None
        if ctx.needs_input_grad[2]:
            dA = s * (DYB.t() @ X)  # [r, K]
        if ctx.needs_input_grad[3]:
            XA = X @ A.t()  # [M, r]
            dB = s * (dY.t() @ XA)  # [N, r]

        return dX, None, dA, dB, None


def lora_dense(X, W, A, B, scaling):
    """Differentiable fused LoRA linear. X may be [*, K]; W=[N,K], A=[r,K], B=[N,r].

    PEFT keeps adapters in fp32 while the base is bf16; cast them to X's dtype
    (differentiable, so grads still reach the fp32 params)."""
    flat = X.reshape(-1, X.shape[-1]) if X.dim() != 2 else X
    A, B = A.to(flat.dtype), B.to(flat.dtype)
    Y = LoRADense.apply(flat, W, A, B, scaling)
    return Y if X.dim() == 2 else Y.reshape(*X.shape[:-1], W.shape[0])
