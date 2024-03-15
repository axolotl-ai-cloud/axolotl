"""
Adapted from:
https://github.com/shawntan/scattermoe
https://arxiv.org/abs/2403.08245
"""

import torch
from torch import nn

from axolotl.monkeypatch.moe import ops
from axolotl.monkeypatch.moe.linear import ParallelExperts


class FusedExperts(nn.Module):
    def __init__(
        self,
        experts,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        activation=nn.SiLU(),
    ):
        """
        This implements fused experts that are compatible with Mixtral.
        MLP of type Gated-Linear Unit, typically with a SiLU activation function.
        """
        super(FusedExperts, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, 2 * hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        # parallelize all w1 and w3 computation by concat + stack
        self.experts.weight = torch.stack(
            [
                torch.cat([experts[i].w1, experts[i].w3], dim=1)
                for i in range(len(experts))
            ],
            dim=0,
            device=experts[0].w1.weight.device,
        )

        # parallelize all w2 computation by stack
        self.output_experts.weight = torch.stack(
            [expert.w2 for expert in experts],
            dim=0,
            device=experts[0].w2.weight.device,
        )

    def forward(
        self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor
    ):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = ops.flatten_and_sort(
                expert_idxs
            )
            padded_block_idxs, expert_offsets = ops.padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

        h, gates = self.experts(
            x,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True,
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        y = self.output_experts(
            h,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_in=True,
            gates=expert_p,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y
