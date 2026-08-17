# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SonicMoE + multi-adapter LoRA via combined-group weight materialization.

SonicMoE dispatches opaque CUTLASS/CuteDSL grouped GEMMs that cannot fuse LoRA, so
the single-adapter path materializes ``W_eff[e] = W[e] + scaling * (B_e @ A_e)``
per expert and hands that to the kernel (see ``lora.MoELoRAMaterialize``).

Multi-tenant co-training keeps that strategy but over the combined
``(expert, tenant)`` grouping: a token carries the router's expert id *and* a
per-row tenant id, and the effective weight for ``(token routed to e, tenant t)``
is ``W[e] + scaling_t * (B_{e,t} @ A_{e,t})``. We materialize all ``E*T`` effective
weights once and remap each token-slot's expert id to ``e * T + t``; the *same*
opaque grouped GEMM, called with ``num_experts = E*T``, then routes every tenant
in a single launch. The frozen base ``W[e]`` is shared in storage but, because the
kernel needs one contiguous weight per group, is broadcast into the ``E*T`` buffer
-- so transient materialization memory grows ~linearly with ``T`` (at ``T == 1``
it is identical to the single-adapter path).

This module owns only the materialize + id-remap (the LoRA-specific part); the
CUTLASS GEMM is unchanged. ``T == 1`` reduces to ``MoELoRAMaterialize`` exactly.
"""

from __future__ import annotations

import torch


class MoEMultiLoRAMaterialize(torch.autograd.Function):
    """Build ``W_eff[e*T + t] = W[e] + scaling_t * (B_{e,t} @ A_{e,t})``.

    Layout (stacked over tenants, outer expert / inner tenant to match the
    ``combined = e*T + t`` group id):
      base    ``[E, out, in]``   (frozen)
      lora_A  ``[T, E, r, in]``
      lora_B  ``[T, E, out, r]``
      scaling ``[T]``
    Returns ``W_eff`` ``[E*T, out, in]``. Gradients flow to A/B only.
    """

    @staticmethod
    def forward(ctx, base_weight, lora_A, lora_B, scaling):
        T, E, r, in_dim = lora_A.shape
        out_dim = base_weight.shape[1]
        if base_weight.dtype != lora_A.dtype:
            base_weight = base_weight.to(lora_A.dtype)

        # Materialize directly in combined (e*T + t) order so the result needs no
        # reorder copy: einsum over the [E, T, ...] view yields a contiguous
        # [E, T, out, in] that reshapes to [E*T, ...] for free. The A/B permutes
        # are on rank-sized tensors (~in/r smaller than W_eff), so ~free. delta is
        # recomputed in backward, so the scale + frozen-base add are in-place.
        a_et = lora_A.permute(1, 0, 2, 3)  # [E, T, r, in]
        b_et = lora_B.permute(1, 0, 2, 3)  # [E, T, out, r]
        delta = torch.einsum("etor,etri->etoi", b_et, a_et)  # [E, T, out, in]
        delta.mul_(scaling.view(1, T, 1, 1))
        delta.add_(base_weight.unsqueeze(1))  # broadcast frozen base over tenants
        w_eff = delta.reshape(E * T, out_dim, in_dim)

        ctx.save_for_backward(lora_A, lora_B, scaling)
        ctx.shape = (T, E, r, in_dim, out_dim)
        return w_eff

    @staticmethod
    def backward(ctx, grad_w_eff):
        lora_A, lora_B, scaling = ctx.saved_tensors
        T, E, r, in_dim, out_dim = ctx.shape
        # Scale the (rank-sized) A/B grads, not the W_eff-sized incoming grad, to
        # avoid a full-size copy in backward.
        g = grad_w_eff.reshape(E, T, out_dim, in_dim)
        a_et = lora_A.permute(1, 0, 2, 3)  # [E, T, r, in]
        b_et = lora_B.permute(1, 0, 2, 3)  # [E, T, out, r]
        # dA = B^T @ dDelta ; dB = dDelta @ A^T (then back to the [T, E, ...] layout)
        d_a_et = torch.einsum("etor,etoi->etri", b_et, g).mul_(scaling.view(1, T, 1, 1))
        d_b_et = torch.einsum("etoi,etri->etor", g, a_et).mul_(scaling.view(1, T, 1, 1))
        return None, d_a_et.permute(1, 0, 2, 3), d_b_et.permute(1, 0, 2, 3), None


def materialize_multi_lora_experts(base_weight, lora_A, lora_B, scaling):
    """Autograd-backed ``W_eff`` builder; see :class:`MoEMultiLoRAMaterialize`."""
    return MoEMultiLoRAMaterialize.apply(base_weight, lora_A, lora_B, scaling)


def combined_expert_ids(expert_ids, tenant_ids, token_idx, num_tenants):
    """Remap each token-slot's expert id to its combined ``e * T + t`` group.

    ``expert_ids`` / ``token_idx`` are the per-(token, top-k slot) tensors sonicmoe
    already builds; ``tenant_ids`` is per *token*. Returns int32 combined ids
    aligned with ``expert_ids`` for a ``num_experts = E*T`` grouped GEMM.
    """
    t = tenant_ids.to(torch.int64)[token_idx.to(torch.int64)]
    return (expert_ids.to(torch.int64) * num_tenants + t).to(torch.int32)
