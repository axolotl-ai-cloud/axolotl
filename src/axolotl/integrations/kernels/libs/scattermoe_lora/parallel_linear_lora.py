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

from typing import Optional

import torch

from .kernels import ops as base_ops
from .kernels.lora_ops import (
    group_bwd_lora,
    group_bwd_lora_fused,
    scatter2scatter_lora,
    scatter2scatter_lora_dX,
)


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
        expert_weights: torch.Tensor,
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
        with torch.device(x.device):
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
            ctx.use_fused_dX = use_fused_dX
            ctx.use_fused_gather = use_fused_gather

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
            N_dim = expert_weights.size(-1)
            fuse_gather_workload = M_total * max(K_dim, N_dim)
            _FUSE_GATHER_THRESHOLD = 2**24  # ~16M elements

            can_fuse_gather = (
                ctx.use_fused_gather
                and not grouped_in  # X must be ungrouped for scatter access
                and gates is None  # gate coeff requires multiplicative gather
                and fuse_gather_workload < _FUSE_GATHER_THRESHOLD
            )

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

                d_lora_A, d_lora_B = group_bwd_lora(
                    DY=grouped_grad_out,
                    X=grouped_x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    expert_offsets=expert_offsets,
                    E=E,
                    scaling=scaling,
                )

            # ------------------------------------------------------------------
            # Input gradient: dX = dY @ W^T + scaling * (dY @ B) @ A
            # ------------------------------------------------------------------
            if ctx.use_fused_dX:
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
                )

                # LoRA part: dX_lora = scaling * (dY @ B) @ A
                if scaling != 0.0:
                    d_input_lora_grouped = _compute_lora_input_grad(
                        grouped_grad_out,
                        lora_A,
                        lora_B,
                        expert_offsets,
                        E,
                        scaling,
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

            # W is frozen during LoRA training -- skip weight gradient
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
) -> torch.Tensor:
    """
    Compute the LoRA contribution to the input gradient:
      dX_lora = scaling * (dY @ B) @ A

    Uses PyTorch ops on expert-grouped data.
    Each expert e: dX_e = scaling * (dY_e @ B_e) @ A_e
    """
    R = lora_A.size(0) // E
    K = lora_A.size(1)
    M_total = grouped_grad_out.size(0)

    d_input_lora = torch.zeros(
        (M_total, K), device=grouped_grad_out.device, dtype=grouped_grad_out.dtype
    )

    compute_dtype = grouped_grad_out.dtype

    prev_offset = 0
    for e in range(E):
        curr_offset = expert_offsets[e].item()
        if curr_offset > prev_offset:
            dy_e = grouped_grad_out[prev_offset:curr_offset]  # [M_e, N]
            a_e = lora_A[e * R : (e + 1) * R, :].to(compute_dtype)  # [r, K]
            b_e = lora_B[:, e * R : (e + 1) * R].to(compute_dtype)  # [N, r]

            # dX_e = scaling * (dY_e @ B_e) @ A_e
            dy_b = dy_e @ b_e  # [M_e, r]
            dx_e = scaling * (dy_b @ a_e)  # [M_e, K]
            d_input_lora[prev_offset:curr_offset] = dx_e

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
