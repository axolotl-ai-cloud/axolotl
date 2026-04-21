# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Pure tensor layout helpers for ScatterMoE LoRA weights."""


def peft_lora_B_to_scattermoe(peft_B, num_experts, rank):
    """Convert peft rank-major lora_B ``[out, E*r]`` to scattermoe
    expert-major ``[N, r*E]``.

    peft reshapes B to ``[out, r, E]`` (rank-major).
    scattermoe slices B as ``[:, e*r:(e+1)*r]`` (expert-major).
    """
    N = peft_B.shape[0]
    return (
        peft_B.reshape(N, rank, num_experts)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(N, num_experts * rank)
    )


def peft_lora_to_scattermoe(peft_A, peft_B, num_experts, rank):
    """Convert peft LoRA weights to scattermoe layout.

    peft operates on the parameter in its native storage layout ``[E, dim1, dim2]``
    where ``out_features=dim1, in_features=dim2``. ScatterMoE transposes the
    parameter (``W = param.transpose(2, 1)``), giving ``[E, dim2, dim1]`` with
    ``K=dim2, N=dim1``.

    peft gives:
        lora_A ``[r*E, dim2]``, lora_B ``[dim1, r*E]``

    scattermoe needs:
        lora_A ``[r*E, K=dim2]``, lora_B ``[N=dim1, r*E]``

    peft's A already matches ScatterMoE's A shape. Only B needs conversion from
    peft's rank-major layout to ScatterMoE's expert-major layout.
    """
    smoe_A = peft_A
    smoe_B = peft_lora_B_to_scattermoe(peft_B, num_experts, rank)

    return smoe_A, smoe_B


def peft_down_proj_lora_to_scattermoe(peft_A, peft_B, num_experts, rank):
    """Deprecated alias for :func:`peft_lora_to_scattermoe`."""
    return peft_lora_to_scattermoe(peft_A, peft_B, num_experts, rank)


def validate_scattermoe_lora_shapes(expert_weights, lora_A, lora_B):
    """Validate LoRA tensor layout before dispatching ScatterMoE kernels."""
    E, K, N = expert_weights.shape
    if lora_A.dim() != 2 or lora_B.dim() != 2:
        raise ValueError(
            "ScatterMoE LoRA expects 2D lora_A and lora_B tensors, got "
            f"lora_A={tuple(lora_A.shape)} and lora_B={tuple(lora_B.shape)}."
        )

    if lora_A.size(0) % E != 0:
        raise ValueError(
            "ScatterMoE LoRA expects lora_A rows to be divisible by the number "
            f"of experts ({E}), got lora_A={tuple(lora_A.shape)}."
        )

    rank = lora_A.size(0) // E
    expected_A = (E * rank, K)
    expected_B = (N, E * rank)
    if tuple(lora_A.shape) != expected_A or tuple(lora_B.shape) != expected_B:
        raise ValueError(
            "Invalid ScatterMoE LoRA layout for expert_weights "
            f"{tuple(expert_weights.shape)}. Expected lora_A={expected_A} and "
            f"lora_B={expected_B}, got lora_A={tuple(lora_A.shape)} and "
            f"lora_B={tuple(lora_B.shape)}. For PEFT target_parameters, keep "
            "lora_A as [E*r, K] and only convert lora_B from rank-major to "
            "expert-major layout."
        )
