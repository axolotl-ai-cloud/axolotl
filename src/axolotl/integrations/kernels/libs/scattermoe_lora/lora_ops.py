# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
ParallelExperts module with LoRA support.

Provides a drop-in replacement for ScatterMoE's ParallelExperts that
uses the fused LoRA kernel when adapter weights are attached.
"""

from typing import Optional

import torch
import torch.nn as nn

from .parallel_linear_lora import parallel_linear_lora


class ParallelExperts(nn.Module):
    """
    Parallel Experts with fused LoRA support.

    Drop-in replacement for the original ParallelExperts. When LoRA parameters
    are attached via set_lora(), the forward pass uses a fused kernel:
        Y = X @ W + scaling * (X @ A^T) @ B^T
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_size))
        else:
            self.bias = None
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self._lora_A: torch.Tensor | None = None
        self._lora_B: torch.Tensor | None = None
        self._lora_scaling: float | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"input_size={self.input_size}, "
            f"output_size={self.output_size}"
        )

    def set_lora(self, lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float):
        """Attach LoRA parameters for fused computation."""
        self._lora_A = lora_A
        self._lora_B = lora_B
        self._lora_scaling = scaling

    def clear_lora(self):
        """Remove LoRA parameters."""
        self._lora_A = None
        self._lora_B = None
        self._lora_scaling = None

    def forward(
        self,
        inputs: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: Optional[torch.Tensor] = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        return parallel_linear_lora(
            inputs,
            self.weight.permute(0, 2, 1),  # [E, input, output]
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A=self._lora_A,
            lora_B=self._lora_B,
            scaling=self._lora_scaling or 1.0,
            expert_biases=self.bias,
            gates=gates,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )
