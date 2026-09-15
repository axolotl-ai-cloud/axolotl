# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Encapsulated scatter2scatter grouped LoRA ops for the TILE-padded marlin/grouped path.

Replaces per-expert torch._grouped_mm loops in grouped_train.py with single-launch
scatter2scatter calls.  All implementation choices (which kernel variant, dispatch)
live inside this module — the training code calls one function per phase.

Layout bridge: grouped_train uses a TILE-padded expert-sorted buffer A[Mt, K] with
per-TILE expert ids (m_indices) and cumulative padded offsets (offs).  The
scatter2scatter kernels in kernels/ops.py work on the same sorted layout when called
with x_grouped=True / y_grouped=True.

LoRA weight layout (stacked, from _lora_stack in grouped_train):
  A : [E, r, in_K]   (A-adapter)
  B : [E, out_N, r]  (B-adapter)

scatter2scatter weight layout: W[E, in, out].  Mapping:
  W_A = A.permute(0, 2, 1)  -> [E, in_K, r]    for X @ A^T step
  W_B = B.permute(0, 2, 1)  -> [E, r, out_N]   for XA @ B^T step
  W_Bt = B                  -> [E, out_N, r]    for dY @ B step (B already [E, out_N, r])
"""

from __future__ import annotations

import torch

from .kernels.grouped_gram import grouped_lora_weight_grads
from .kernels.ops import scatter2scatter


def _mk_routing(offs: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build scatter2scatter routing tensors from the TILE-padded grouped layout.

    Args:
        offs: cumulative TILE-padded expert offsets [E] (int32), as produced by
              grouped_fp4_moe_train — offs[e] = sum of tile-padded row counts
              for experts 0..e.
        E: number of experts.

    Returns:
        (sorted_expert_idxs [Mt], sorted_scattered_idxs [Mt])
        sorted_expert_idxs[i] = expert id for row i of the grouped buffer.
        sorted_scattered_idxs = identity [0..Mt-1]; with x_grouped=y_grouped=True
        the scatter2scatter kernel reads/writes M_block directly and ignores it
        for X/Y addressing, but it must have the right length.
    """
    Mt = int(offs[-1].item())
    dev = offs.device
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    sizes = offs - starts  # per-expert padded row count
    e_ids = torch.arange(E, device=dev, dtype=torch.int32)
    sorted_expert_idxs = e_ids.repeat_interleave(sizes)
    sorted_scattered_idxs = torch.arange(Mt, device=dev, dtype=torch.int32)
    return sorted_expert_idxs, sorted_scattered_idxs


def grouped_lora_fwd(
    x: torch.Tensor,  # [Mt, in_K], expert-grouped (TILE-padded)
    A: torch.Tensor,  # [E, r, in_K]
    B: torch.Tensor,  # [E, out_N, r]
    scaling: float,
    offs: torch.Tensor,  # [E] int32 cumulative TILE-padded offsets
    E: int,
    residual: torch.Tensor
    | None = None,  # [Mt, out_N] base output to fold into the epilogue
) -> tuple[torch.Tensor, torch.Tensor]:
    """LoRA forward on the TILE-padded grouped layout.

    Computes Y_lora = scaling * (X @ A^T) @ B^T, row-for-row with the marlin
    base output.  Two scatter2scatter launches with x_grouped=y_grouped=True keep
    the result in the padded grouped layout for direct in-place addition to base.

    If ``residual`` is given (the base expert GEMM output), it is added in the LoRA-B GEMM epilogue,
    so the returned tensor is ``base + scaling*lora`` with NO separate add pass or temp tensor.

    Returns:
        (y [Mt, out_N], xa [Mt, r])  — y = (residual + scaling*lora) if residual else scaling*lora;
        xa is saved (unscaled) for the backward dB/dA.
    """
    sei, ssi = _mk_routing(offs, E)

    W_A = A.permute(0, 2, 1).contiguous()  # [E, in_K, r]
    W_B = B.permute(0, 2, 1).contiguous()  # [E, r, out_N]

    xa = scatter2scatter(
        X=x,
        W=W_A,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=1,
        x_grouped=True,
        y_grouped=True,
    )  # [Mt, r]

    # Scale the tiny [Mt, r] inner activation, not the large [Mt, out_N] output (identical result,
    # far fewer elements). xa is returned UNSCALED; the backward derives scaling separately.
    y_lora = scatter2scatter(
        X=xa * scaling,
        W=W_B,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=1,
        x_grouped=True,
        y_grouped=True,
        residual=residual,
    )  # [Mt, out_N], = (residual + scaling*lora) if residual else scaling*lora
    return y_lora, xa


def grouped_lora_bwd(
    dy: torch.Tensor,  # [Mt, out_N], grad w.r.t. LoRA output, grouped
    x: torch.Tensor,  # [Mt, in_K], forward input, grouped
    A: torch.Tensor,  # [E, r, in_K]
    B: torch.Tensor,  # [E, out_N, r]
    xa: torch.Tensor,  # [Mt, r], X@A^T from forward (saved)
    scaling: float,
    offs: torch.Tensor,  # [E] int32 cumulative TILE-padded offsets
    E: int,
    residual: torch.Tensor
    | None = None,  # [Mt, in_K] base dX to fold into the dX_lora epilogue
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LoRA backward on the TILE-padded grouped layout.

    Computes:
        dX_lora = scaling * (dY @ B) @ A   [Mt, in_K]
        dA                                  [E, r, in_K]
        dB                                  [E, out_N, r]

    If ``residual`` is given (the base dX), it is added in the dX_lora GEMM epilogue, so the returned
    dx is ``base_dX + scaling*lora_dX`` with no separate add pass.

    Returns:
        (dx, dA, dB)  — dx = (residual + scaling*lora_dX) if residual else scaling*lora_dX
    """
    sei, ssi = _mk_routing(offs, E)

    r = A.size(1)
    in_K = A.size(2)
    out_N = B.size(1)

    # yb = dY @ B; B is already [E, out_N, r] = [E, K=out_N, N=r]
    yb = scatter2scatter(
        X=dy,
        W=B.contiguous(),
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=1,
        x_grouped=True,
        y_grouped=True,
    )  # [Mt, r]

    # dx_lora = scaling * yb @ A; A is [E, r, in_K] = [E, K=r, N=in_K]. Fold `scaling` into the tiny
    # [Mt, r] yb, not the large [Mt, in_K] output (identical result, far fewer elements).
    dx_lora = scatter2scatter(
        X=yb * scaling,
        W=A.contiguous(),
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=1,
        x_grouped=True,
        y_grouped=True,
        residual=residual,
    )  # [Mt, in_K], = (residual + scaling*lora) if residual else scaling*lora

    # grouped-Gram kernel expects flat scattermoe layout: lora_A [r*E, in_K], lora_B [out_N, r*E]
    lora_A_flat = A.reshape(E * r, in_K)
    lora_B_flat = B.permute(1, 0, 2).reshape(out_N, E * r)

    dA_flat, dB_flat = grouped_lora_weight_grads(
        grouped_grad_out=dy,
        grouped_x=x,
        yb=yb,
        xa=xa,
        lora_A=lora_A_flat,
        lora_B=lora_B_flat,
        combined_offsets=offs,
        e_total=E,
        scaling=scaling,
    )
    dA = dA_flat.reshape(E, r, in_K)
    dB = dB_flat.reshape(out_N, E, r).permute(1, 0, 2).contiguous()

    return dx_lora, dA, dB
