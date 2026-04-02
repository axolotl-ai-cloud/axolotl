# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
SonicMoE LoRA support via runtime weight materialization.

SonicMoE uses opaque CUTLASS kernels that cannot be modified to fuse LoRA.
Instead, we materialize the effective weight W_eff = W + scaling * (B @ A)
before each CUTLASS call, and use a custom autograd.Function to route
gradients back to the LoRA A and B parameters.

PEFT unwrapping utilities are also provided to handle the ParamWrapper
chain that PEFT creates when targeting expert parameters.
"""

from typing import Optional

import torch

# =============================================================================
# PEFT unwrapping utilities
# =============================================================================


def has_lora(module) -> bool:
    """Check if a module is wrapped by PEFT with LoRA."""
    return hasattr(module, "base_layer") and hasattr(module, "lora_A")


def get_lora_params_from_wrapper(module) -> tuple:
    """Extract LoRA parameters from a PEFT ParamWrapper.

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

    if (
        adapter_name not in lora_A_dict
        or adapter_name not in lora_B_dict
        or adapter_name not in scaling_dict
    ):
        return None, None, None

    lora_A = lora_A_dict[adapter_name].weight
    lora_B = lora_B_dict[adapter_name].weight
    scaling = scaling_dict[adapter_name]

    return lora_A, lora_B, scaling


def unwrap_gate_lora(gate_module):
    """Unwrap PEFT ParamWrapper on the router gate.

    When PEFT targets ``gate.weight``, ``self.gate`` becomes::

        ParamWrapper(weight)
          -> base_layer: Router (the real module)

    Returns:
        (base_gate, gate_weight, gate_lora_delta_or_None)

        ``base_gate`` is the original router module (with ``.top_k``, etc.).
        ``gate_weight`` is the base router weight tensor.
        ``gate_lora_delta_or_None`` is the LoRA delta if active, else None.
        Kept separate to avoid mixing DTensor + Tensor under FSDP.
    """
    if has_lora(gate_module):
        base_gate = gate_module.base_layer
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(gate_module)
        if lora_A is not None:
            delta = scaling * (lora_B @ lora_A)
            return base_gate, base_gate.weight, delta
        return base_gate, base_gate.weight, None

    return gate_module, gate_module.weight, None


def unwrap_experts_lora(experts_module):
    """Walk a PEFT ParamWrapper chain on ``self.experts``.

    When PEFT targets ``experts.gate_up_proj`` and ``experts.down_proj``
    via ``target_parameters``, ``self.experts`` becomes::

        ParamWrapper(down_proj)
          -> base_layer: ParamWrapper(gate_up_proj)
              -> base_layer: Experts (the real module)

    Returns:
        (base_experts, lora_dict)

        ``lora_dict`` maps parameter names to ``(lora_A, lora_B, scaling)``
        tuples, or is empty if no LoRA is active.
    """
    wrappers = {}
    module = experts_module
    while hasattr(module, "base_layer") and hasattr(module, "lora_A"):
        param_name = getattr(module, "parameter_name", None)
        if param_name is not None:
            wrappers[param_name] = module
        module = module.base_layer

    base_experts = module
    lora_dict = {}

    for param_name, wrapper in wrappers.items():
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(wrapper)
        if lora_A is not None:
            lora_dict[param_name] = (lora_A, lora_B, scaling)

    return base_experts, lora_dict


# =============================================================================
# LoRA weight materialization autograd function
# =============================================================================


class MoELoRAMaterialize(torch.autograd.Function):
    """Materialize effective weight W_eff = W + scaling * (B @ A) per expert.

    Inserts into the autograd graph between PEFT's LoRA parameters and
    SonicMoE's CUTLASS kernels. The CUTLASS backward computes dW_eff,
    which this function decomposes into dA and dB via the chain rule.

    Weight layouts (PEFT rank-major):
        base_weight: [E, dim1, dim2]  (frozen expert parameter)
        lora_A:      [r*E, dim2]      (rows [e*r:(e+1)*r] = A_e)
        lora_B:      [dim1, r*E]      (cols [:, e*r:(e+1)*r] = B_e)

    Per-expert: delta_e = B_e @ A_e = [dim1, r] @ [r, dim2] = [dim1, dim2]
    """

    @staticmethod
    def forward(
        ctx,
        base_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        E, dim1, dim2 = base_weight.shape
        r = lora_A.shape[0] // E
        assert lora_A.shape[0] == r * E, (
            f"lora_A rows ({lora_A.shape[0]}) must be divisible by num_experts ({E})"
        )

        # Reshape PEFT rank-major to per-expert batched format
        A_3d = lora_A.reshape(E, r, dim2)
        B_3d = lora_B.reshape(dim1, r, E).permute(2, 0, 1).contiguous()  # [E, dim1, r]

        # Batched matmul: [E, dim1, r] @ [E, r, dim2] = [E, dim1, dim2]
        delta = torch.bmm(B_3d, A_3d)

        W_eff = base_weight + scaling * delta

        ctx.save_for_backward(lora_A, lora_B)
        ctx.scaling = scaling
        ctx.E = E
        ctx.r = r

        return W_eff

    @staticmethod
    def backward(ctx, grad_W_eff: torch.Tensor):
        lora_A, lora_B = ctx.saved_tensors
        scaling = ctx.scaling
        E = ctx.E
        r = ctx.r

        _, dim1, dim2 = grad_W_eff.shape

        # Reshape to per-expert (same as forward)
        A_3d = lora_A.reshape(E, r, dim2)
        B_3d = lora_B.reshape(dim1, r, E).permute(2, 0, 1).contiguous()  # [E, dim1, r]

        # dA_e = scaling * B_e^T @ dW_e
        # [E, r, dim1] @ [E, dim1, dim2] = [E, r, dim2]
        d_A_3d = scaling * torch.bmm(B_3d.transpose(1, 2), grad_W_eff)

        # dB_e = scaling * dW_e @ A_e^T
        # [E, dim1, dim2] @ [E, dim2, r] = [E, dim1, r]
        d_B_3d = scaling * torch.bmm(grad_W_eff, A_3d.transpose(1, 2))

        # Reshape back to PEFT rank-major layout
        d_lora_A = d_A_3d.reshape(E * r, dim2)
        d_lora_B = d_B_3d.permute(1, 2, 0).contiguous().reshape(dim1, E * r)

        return None, d_lora_A, d_lora_B, None


def materialize_expert_lora(
    base_weight: torch.Tensor,
    lora_params: Optional[tuple],
) -> torch.Tensor:
    """Materialize effective expert weight with optional LoRA delta.

    Args:
        base_weight: [E, dim1, dim2] frozen expert parameter
        lora_params: (lora_A, lora_B, scaling) or None

    Returns:
        W_eff if lora_params is not None, else base_weight unchanged.
    """
    if lora_params is None:
        return base_weight
    lora_A, lora_B, scaling = lora_params
    return MoELoRAMaterialize.apply(base_weight, lora_A, lora_B, scaling)
