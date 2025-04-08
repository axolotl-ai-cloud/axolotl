"""
Modified Llama-4 text experts modeling for linearized experts for improved LoRA support
"""

import sys

import torch
from torch import nn
from transformers import Llama4Config
from transformers.activations import ACT2FN


class Llama4TextExperts(nn.Module):
    """
    Modified Llama-4 text experts modeling for linearized experts
    """

    def __init__(self, config: Llama4Config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # Replace fused gate_up_proj with separate Linear modules
        self.gate_projs = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.expert_dim, bias=False)
                for _ in range(self.num_experts)
            ]
        )

        self.up_projs = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.expert_dim, bias=False)
                for _ in range(self.num_experts)
            ]
        )

        # Replace down_proj Parameter with Linear modules
        self.down_projs = nn.ModuleList(
            [
                nn.Linear(self.expert_dim, self.hidden_size, bias=False)
                for _ in range(self.num_experts)
            ]
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward method using separate Linear layers for each expert.

        Args:
            hidden_states (torch.Tensor): (num_experts * batch_size, hidden_size)
                The input should be organized by expert

        Returns:
            torch.Tensor: (num_experts * batch_size, hidden_size)
        """
        # Reshape to separate by expert
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        # batch_size_per_expert = hidden_states.size(1)

        # Initialize output tensor
        next_states = torch.zeros_like(hidden_states)

        # Process each expert separately
        for i in range(self.num_experts):
            # Get input for this expert
            expert_input = hidden_states[
                i
            ]  # Shape: (batch_size_per_expert, hidden_size)

            # Apply gate and up projections
            gate = self.gate_projs[i](
                expert_input
            )  # Shape: (batch_size_per_expert, expert_dim)
            up = self.up_projs[i](
                expert_input
            )  # Shape: (batch_size_per_expert, expert_dim)

            # Apply activation and down projection
            next_states[i] = self.down_projs[i](up * self.act_fn(gate))

        # Flatten back to original shape
        return next_states.view(-1, self.hidden_size)


def patch_llama4_linearized_modeling():
    """
    Patch Llama4TextExperts to use separate Linear layers for each expert.
    """
    from transformers.models.llama4 import modeling_llama4

    modeling_llama4.Llama4TextExperts = Llama4TextExperts
    setattr(
        sys.modules["transformers.models.llama4"],
        "Llama4TextExperts",
        Llama4TextExperts,
    )
