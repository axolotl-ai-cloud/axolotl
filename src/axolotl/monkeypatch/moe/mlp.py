"""
Adapted from:
https://github.com/shawntan/scattermoe
https://arxiv.org/abs/2403.08245
"""

import gc
import torch
from torch import nn

from axolotl.monkeypatch.moe import ops
from axolotl.monkeypatch.moe.linear import ParallelExperts


class FusedExperts(nn.Module):
    def __init__(
        self,
        experts: nn.ModuleList =None,
        hidden_dim=128,
        ffn_dim=512,
        num_experts=8,
        top_k=2,
        activation=nn.SiLU(),
    ):
        """
        This implements fused experts that are compatible with Mixtral.
        MLP of type Gated-Linear Unit, typically with a SiLU activation function.
        """
        super(FusedExperts, self).__init__()

        device = experts[0].w1.weight.device
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.experts = ParallelExperts(num_experts, hidden_dim, 2 * ffn_dim, device=device)
        self.output_experts = ParallelExperts(num_experts, ffn_dim, hidden_dim, device=device)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        with torch.no_grad():
            for i in range(len(experts)):
                self.experts.weight.data[i].copy_(
                    torch.cat(
                        [experts[i].w1.weight.detach(), experts[i].w3.weight.detach()],
                        dim=0
                    )
                )
                self.output_experts.weight.data[i].copy_(
                    experts[i].w2.weight.detach()
                )

    def forward(
        self, x: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = ops.flatten_and_sort(
                selected_experts
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
            gates=routing_weights,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y
