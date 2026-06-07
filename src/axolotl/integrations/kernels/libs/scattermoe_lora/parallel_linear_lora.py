# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
ScatterMoE + LoRA Autograd Function
====================================

Provides the autograd function and Python interface for fused ScatterMoE + LoRA.

Key design for LoRA training:
  - Expert weights W are FROZEN (no gradient computed for W).
  - Only LoRA adapter weights (A, B) receive gradients.
  - The input gradient dX is still computed (needed for upstream layers).
  - This avoids the expensive group_bwd_W computation entirely.

Forward:
  Y = X @ W + scaling * (X @ A^T) @ B^T

Backward (W frozen):
  dX = dY @ W^T + scaling * (dY @ B) @ A          (via scatter2scatter for base, separate for LoRA)
  dA = scaling * (dY @ B)^T @ X                     (per-expert, on grouped data)
  dB = scaling * dY^T @ (X @ A^T)                   (per-expert, on grouped data)
"""

from typing import Optional, Union

import torch

from .kernels import ops as base_ops
from .kernels.grouped_gram import grouped_lora_weight_grads
from .kernels.lora_ops import (
    group_bwd_lora_fused,
    scatter2scatter_lora,
    scatter2scatter_lora_dX,
    scatter2scatter_lora_dX_mx,
    scatter2scatter_lora_mx,
)
from .mx_weights import MXLayout, MXWeights
from .parallel_experts import _INT_MAX, _needs_int64_indices


class ScatterMoELoRA(torch.autograd.Function):
    """
    Autograd function for fused ScatterMoE + LoRA with frozen expert weights.

    This function is optimized for the LoRA fine-tuning scenario where:
    - Expert weights W are frozen (requires_grad=False)
    - Only LoRA A and B matrices receive gradients
    - Input gradients are computed for upstream layer backprop
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_weights: Union[torch.Tensor, MXWeights],
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        expert_biases: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
        use_fused_dX: bool = False,
        use_fused_gather: bool = False,
    ):
        if isinstance(expert_weights, MXWeights):
            assert expert_weights.layout == MXLayout.FWD, (
                "MXWeights passed to forward must be in FWD layout"
            )
            is_mx = True
        else:
            # Cast weights to match input dtype (e.g. 8-bit LoRA)
            if expert_weights.dtype != x.dtype:
                expert_weights = expert_weights.to(x.dtype)
            is_mx = False
        if expert_biases is not None and expert_biases.dtype != x.dtype:
            expert_biases = expert_biases.to(x.dtype)
        L_scattered = sorted_expert_idxs.size(0)
        if is_mx:
            N_dim = expert_weights.N  # type: ignore[union-attr]
        else:
            N_dim = expert_weights.size(-1)  # type: ignore[union-attr]
        # Forward output is [L_scattered, N]. Overflow risk is dominated by
        # that buffer; also probe X for the unusual case where it alone is
        # huge (e.g. very wide hidden with modest seq).
        needs_int64_fwd = (L_scattered * N_dim) >= _INT_MAX or _needs_int64_indices(x)
        with torch.device(x.device):
            if is_mx:
                # Fused MXFP4 forward: dequant happens inside the K-loop
                output = scatter2scatter_lora_mx(
                    X=x,
                    W_mx=expert_weights,
                    sorted_expert_idxs=sorted_expert_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    k=k,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    b=expert_biases,
                    x_grouped=grouped_in,
                    y_grouped=grouped_out,
                    int64_indices=needs_int64_fwd,
                )
            else:
                # Fused forward: Y = X @ W + scaling * (X @ A^T) @ B^T
                output = scatter2scatter_lora(
                    X=x,
                    W=expert_weights,
                    sorted_expert_idxs=sorted_expert_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    k=k,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    b=expert_biases,
                    x_grouped=grouped_in,
                    y_grouped=grouped_out,
                    int64_indices=needs_int64_fwd,
                )

            # Handle gating (weighted combination of top-k expert outputs)
            if gates is not None:
                output_expanded = output.view(
                    gates.size(0), gates.size(1), output.size(-1)
                )
                output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
            else:
                output_expanded = None

            ctx.save_for_backward(
                x,
                lora_A,
                lora_B,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                gates,
                output_expanded,
            )
            # Store frozen weights as plain Python attributes instead of
            # save_for_backward.  This avoids:
            # 1. Version-check conflicts with FSDP unshard/reshard
            # 2. Pinning all-gathered parameters via saved_tensors hooks
            # 3. Interfering with activation offloading pack/unpack hooks
            # Safe because expert_weights are frozen (requires_grad=False).
            ctx.expert_weights = expert_weights
            ctx.expert_biases = expert_biases
            ctx.grouped_in = grouped_in
            ctx.grouped_out = grouped_out
            ctx.k = k
            ctx.scaling = scaling
            # MXFP4 forces fused dX + gather: the non-fused dX path would have
            # to materialise a bf16 weight tile, defeating the kernel-fusion
            # win, and the gather/scatter pattern is identical.
            ctx.use_fused_dX = True if is_mx else use_fused_dX
            ctx.use_fused_gather = True if is_mx else use_fused_gather
            ctx.is_mx = is_mx

        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        with torch.device(grad_out.device):
            (
                x,
                lora_A,
                lora_B,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                gates,
                output_expanded,
            ) = ctx.saved_tensors
            expert_weights = ctx.expert_weights

            k = ctx.k
            scaling = ctx.scaling
            grouped_in = ctx.grouped_in
            grouped_out = ctx.grouped_out
            is_mx = ctx.is_mx
            if is_mx:
                E = expert_weights.packed.size(0)
            else:
                E = expert_weights.size(0)

            # ------------------------------------------------------------------
            # Gate gradients (if using top-k gating with routing weights)
            # ------------------------------------------------------------------
            if gates is not None:
                # d_gates[t, j] = output_expanded[t, j, :] . grad_out[t, :]
                d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)
                # Reuse output_expanded buffer for grouped_grad_out
                grouped_grad_out = output_expanded.flatten(0, 1)
            else:
                d_gates = None
                gates_flat = None
                gate_fan = 1
                grouped_grad_out = None

            # ------------------------------------------------------------------
            # LoRA gradients (dA, dB) and setup for dX
            # ------------------------------------------------------------------
            # Fused gather uses sorted_scattered_idxs for indirect X access
            # in the Triton kernel, avoiding the group(x) allocation.
            #
            # can_fuse_gather: X is ungrouped and not too large for scatter loads
            #   - When gates is None and grouped_out=False: both DY and X ungrouped
            #   - When grouped_out=True (gate_up_proj): DY already grouped, X ungrouped
            #     -> use dy_grouped=True in the fused kernel
            M_total = sorted_scattered_idxs.size(0)
            K_dim = x.size(-1)
            N_dim = expert_weights.N if is_mx else expert_weights.size(-1)
            fuse_gather_workload = M_total * max(K_dim, N_dim)
            _FUSE_GATHER_THRESHOLD = 2**24  # ~16M elements

            can_fuse_gather = (
                ctx.use_fused_gather
                and not grouped_in  # X must be ungrouped for scatter access
                and gates is None  # gate coeff requires multiplicative gather
                and fuse_gather_workload < _FUSE_GATHER_THRESHOLD
            )

            # The backward path indexes into grad_out [M_total, N] and x [M, K]
            # using either M_idx (grouped) or scatter_idx (ungrouped). Overflow
            # risk is dominated by the largest indexed buffer along the M axis.
            needs_int64_bwd = (
                (M_total * N_dim) >= _INT_MAX
                or (M_total * K_dim) >= _INT_MAX
                or _needs_int64_indices(grad_out, x)
            )

            yb = None  # dY @ B, computed by the non-fused dA/dB path; reused by dX_lora
            if can_fuse_gather:
                # ------------------------------------------------------------------
                # Fused path: skip group(x) entirely
                # ------------------------------------------------------------------
                d_expanded_input = None

                d_lora_A, d_lora_B = group_bwd_lora_fused(
                    DY=grad_out,
                    X=x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    expert_offsets=expert_offsets,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    E=E,
                    k=k,
                    scaling=scaling,
                    dy_grouped=grouped_out,
                    int64_indices=needs_int64_bwd,
                )

                # Prepare grouped_grad_out for the dX path (needed by both
                # the fused dX kernel when grouped_out=True, and the non-fused path)
                if grouped_out:
                    grouped_grad_out = grad_out
                elif not ctx.use_fused_dX:
                    grouped_grad_out = base_ops.group(
                        grad_out,
                        sorted_scattered_idxs,
                        fan_out=gate_fan,
                        coeff=gates_flat,
                        out=grouped_grad_out,
                    )
            else:
                # ------------------------------------------------------------------
                # Original path: explicit group() calls
                # ------------------------------------------------------------------
                if grouped_out:
                    grouped_grad_out = grad_out
                else:
                    grouped_grad_out = base_ops.group(
                        grad_out,
                        sorted_scattered_idxs,
                        fan_out=gate_fan,
                        coeff=gates_flat,
                        out=grouped_grad_out,
                    )

                if grouped_in:
                    grouped_x = x
                    d_expanded_input = None
                else:
                    grouped_x = base_ops.group(x, sorted_scattered_idxs, fan_out=k)
                    d_expanded_input = grouped_x  # Will be overwritten; reuse buffer

                # dA/dB via grouped-Gram over precomputed XA/YB (rank-sized) instead
                # of the split kernel's per-output-block recompute -- a large win as
                # the expert count grows (modern MoEs, E >= 128). YB is also reused by
                # the non-fused dX path below.
                rank = lora_A.size(0) // E
                k_dim = lora_A.size(1)
                n_dim = lora_B.size(0)
                w_yb = lora_B.reshape(n_dim, E, rank).permute(1, 0, 2).contiguous()
                yb = base_ops.scatter2scatter(
                    X=grouped_grad_out,
                    W=w_yb,
                    k=1,
                    sorted_expert_idxs=sorted_expert_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    x_grouped=True,
                    y_grouped=True,
                    int64_indices=needs_int64_bwd,
                )
                w_xa = lora_A.reshape(E, rank, k_dim).permute(0, 2, 1).contiguous()
                xa = base_ops.scatter2scatter(
                    X=grouped_x,
                    W=w_xa,
                    k=1,
                    sorted_expert_idxs=sorted_expert_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    x_grouped=True,
                    y_grouped=True,
                    int64_indices=needs_int64_bwd,
                )
                d_lora_A, d_lora_B = grouped_lora_weight_grads(
                    grouped_grad_out,
                    grouped_x,
                    yb,
                    xa,
                    lora_A,
                    lora_B,
                    expert_offsets,
                    E,
                    scaling,
                )

            # ------------------------------------------------------------------
            # Input gradient: dX = dY @ W^T + scaling * (dY @ B) @ A
            # ------------------------------------------------------------------
            if is_mx:
                # dX kernel reuses the forward MX layout (block axis = K) —
                # no pre-transpose/re-quantize needed.
                if can_fuse_gather and not grouped_out:
                    d_expanded_input = scatter2scatter_lora_dX_mx(
                        DY=grad_out,
                        W_mx=expert_weights,
                        sorted_expert_idxs=sorted_expert_idxs,
                        sorted_scattered_idxs=sorted_scattered_idxs,
                        k=1,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        dy_grouped=False,
                        dx_grouped=grouped_in,
                        out=d_expanded_input,
                        int64_indices=needs_int64_bwd,
                    )
                else:
                    d_expanded_input = scatter2scatter_lora_dX_mx(
                        DY=grouped_grad_out,
                        W_mx=expert_weights,
                        sorted_expert_idxs=sorted_expert_idxs,
                        sorted_scattered_idxs=sorted_scattered_idxs,
                        k=1,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        dy_grouped=True,
                        dx_grouped=grouped_in,
                        out=d_expanded_input,
                        int64_indices=needs_int64_bwd,
                    )
            elif ctx.use_fused_dX:
                if can_fuse_gather and not grouped_out:
                    # Fully fused: read ungrouped DY via scatter pattern
                    d_expanded_input = scatter2scatter_lora_dX(
                        DY=grad_out,
                        W=expert_weights,
                        sorted_expert_idxs=sorted_expert_idxs,
                        sorted_scattered_idxs=sorted_scattered_idxs,
                        k=1,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        dy_grouped=False,
                        dx_grouped=grouped_in,
                        out=d_expanded_input,
                        int64_indices=needs_int64_bwd,
                    )
                else:
                    # Fused dX only: read from pre-grouped DY
                    d_expanded_input = scatter2scatter_lora_dX(
                        DY=grouped_grad_out,
                        W=expert_weights,
                        sorted_expert_idxs=sorted_expert_idxs,
                        sorted_scattered_idxs=sorted_scattered_idxs,
                        k=1,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        dy_grouped=True,
                        dx_grouped=grouped_in,
                        out=d_expanded_input,
                        int64_indices=needs_int64_bwd,
                    )
            else:
                # Original path: separate base scatter2scatter + LoRA Python loop
                d_expanded_input = base_ops.scatter2scatter(
                    X=grouped_grad_out,
                    x_grouped=True,
                    W=expert_weights.permute(0, 2, 1),  # [E, N, K]
                    sorted_expert_idxs=sorted_expert_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    k=1,
                    y_grouped=grouped_in,
                    out=d_expanded_input,
                    int64_indices=needs_int64_bwd,
                )

                # LoRA part: dX_lora = scaling * (dY @ B) @ A (sync-free grouped GEMMs;
                # reuses YB from the dA/dB path when it was computed there)
                if scaling != 0.0:
                    d_input_lora_grouped = _compute_lora_input_grad(
                        grouped_grad_out,
                        lora_A,
                        lora_B,
                        expert_offsets,
                        E,
                        scaling,
                        sorted_expert_idxs=sorted_expert_idxs,
                        sorted_scattered_idxs=sorted_scattered_idxs,
                        int64_indices=needs_int64_bwd,
                        yb=yb,
                    )
                    if grouped_in:
                        d_expanded_input.add_(d_input_lora_grouped)
                    else:
                        # Scatter-add LoRA gradient directly into d_expanded_input.
                        # Avoids allocating a zeros_like + add result
                        d_expanded_input[sorted_scattered_idxs] += d_input_lora_grouped

            # Reduce over top-k if k > 1
            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(
                    x.size(0), k, d_expanded_input.size(-1)
                ).sum(-2)

            # W is frozen during LoRA training -- skip weight gradient.
            # (MX weights are containers, not tensors, and never carry grad.)
            if is_mx:
                d_weights = None
            else:
                d_weights = (
                    torch.zeros_like(expert_weights)
                    if expert_weights.requires_grad
                    else None
                )
            d_biases = None

        return (
            d_input,
            d_weights,
            None,
            None,
            None,
            None,  # k, sorted indices, offsets
            d_lora_A,
            d_lora_B,
            None,  # lora_A, lora_B, scaling
            d_biases,
            d_gates,
            None,
            None,  # grouped_in, grouped_out
            None,  # use_fused_dX
            None,  # use_fused_gather
        )


def _compute_lora_input_grad(
    grouped_grad_out: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_offsets: torch.Tensor,
    E: int,
    scaling: float,
    sorted_expert_idxs: Optional[torch.Tensor] = None,
    sorted_scattered_idxs: Optional[torch.Tensor] = None,
    int64_indices: bool = False,
    yb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LoRA contribution to the input gradient: ``dX_lora = scaling * (dY @ B) @ A``,
    on expert-grouped data.

    With routing ids it runs as two grouped GEMMs (``scatter2scatter``) -- sync-free
    and a single launch each, instead of a Python per-expert loop with an
    ``expert_offsets[e].item()`` device sync per expert (O(E) syncs, which dominate
    at the high expert counts of modern MoEs). ``yb = dY @ B`` may be passed in to
    reuse the value already computed for the dA/dB grads. The per-expert loop is kept
    as a fallback for callers without the routing ids.
    """
    R = lora_A.size(0) // E
    K = lora_A.size(1)
    N = lora_B.size(0)

    if sorted_expert_idxs is not None:
        if yb is None:
            w_yb = lora_B.reshape(N, E, R).permute(1, 0, 2).contiguous()  # [E, N, R]
            yb = base_ops.scatter2scatter(
                X=grouped_grad_out,
                W=w_yb,
                k=1,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                x_grouped=True,
                y_grouped=True,
                int64_indices=int64_indices,
            )
        w_a = lora_A.reshape(E, R, K).contiguous()  # [E, R, K]
        dx = base_ops.scatter2scatter(
            X=yb,
            W=w_a,
            k=1,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            x_grouped=True,
            y_grouped=True,
            int64_indices=int64_indices,
        )
        return dx.mul_(scaling)

    # fallback (no routing ids): one host sync for the whole offset array, not per expert
    offsets = expert_offsets.tolist()
    compute_dtype = grouped_grad_out.dtype
    d_input_lora = torch.zeros(
        (grouped_grad_out.size(0), K),
        device=grouped_grad_out.device,
        dtype=compute_dtype,
    )
    prev_offset = 0
    for e in range(E):
        curr_offset = offsets[e]
        if curr_offset > prev_offset:
            dy_e = grouped_grad_out[prev_offset:curr_offset]
            a_e = lora_A[e * R : (e + 1) * R, :].to(compute_dtype)
            b_e = lora_B[:, e * R : (e + 1) * R].to(compute_dtype)
            d_input_lora[prev_offset:curr_offset] = scaling * ((dy_e @ b_e) @ a_e)
        prev_offset = curr_offset
    return d_input_lora


# =============================================================================
# Helper: Extract LoRA params from PEFT ParamWrapper
# =============================================================================


def get_lora_params_from_wrapper(module) -> tuple:
    """
    Extract LoRA parameters from a PEFT ParamWrapper.

    Returns:
        (lora_A, lora_B, scaling) if LoRA is active, else (None, None, None)
    """
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return None, None, None

    active_adapters = getattr(module, "active_adapters", ["default"])
    if not active_adapters:
        return None, None, None

    adapter_name = active_adapters[0]

    lora_A_dict = getattr(module, "lora_A", {})
    lora_B_dict = getattr(module, "lora_B", {})
    scaling_dict = getattr(module, "scaling", {})

    if adapter_name not in lora_A_dict:
        return None, None, None

    lora_A = lora_A_dict[adapter_name].weight
    lora_B = lora_B_dict[adapter_name].weight
    scaling = scaling_dict[adapter_name]

    return lora_A, lora_B, scaling


# =============================================================================
# Drop-in replacement for parallel_linear
# =============================================================================


def parallel_linear_lora(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    k: int,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    expert_offsets: torch.Tensor,
    lora_A: Optional[torch.Tensor] = None,
    lora_B: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
    expert_biases: Optional[torch.Tensor] = None,
    gates: Optional[torch.Tensor] = None,
    grouped_in: bool = False,
    grouped_out: bool = False,
    use_fused_dX: bool = False,
    use_fused_gather: bool = False,
):
    """
    Drop-in replacement for parallel_linear that supports LoRA.

    If lora_A and lora_B are provided, uses fused LoRA kernel.
    Otherwise falls back to standard scatter2scatter.
    """
    if lora_A is not None and lora_B is not None:
        return ScatterMoELoRA.apply(
            inputs,
            expert_weights,
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A,
            lora_B,
            scaling,
            expert_biases,
            gates,
            grouped_in,
            grouped_out,
            use_fused_dX,
            use_fused_gather,
        )
    else:
        from .parallel_experts import ParallelLinear

        return ParallelLinear.apply(
            inputs,
            expert_weights,
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            expert_biases,
            gates,
            grouped_in,
            grouped_out,
        )
