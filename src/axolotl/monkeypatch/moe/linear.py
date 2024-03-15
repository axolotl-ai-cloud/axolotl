"""
Adapted from:
https://github.com/shawntan/scattermoe
https://arxiv.org/abs/2403.08245
"""

import torch
import torch.nn as nn
from axolotl.monkeypatch.moe import ops

class ParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, expert_weights, k,
        sorted_expert_idxs, sorted_scattered_idxs,
        padded_block_idxs, expert_offsets,
        gates=None, grouped_in=False, grouped_out=False,
    ):

        output = ops.scatter2scatter(
            X=x, W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=k, x_grouped=grouped_in, y_grouped=grouped_out
        )
        if gates is not None:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(
                gates[:, None, :],
                output_expanded
            ).squeeze(1)
        else:
            output_expanded = None

        ctx.save_for_backward(
            x, expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates,
            output_expanded
        )
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
        return output
    @staticmethod
    def backward(ctx, grad_out):
        (x, expert_weights,
         sorted_expert_idxs,
         sorted_scattered_idxs,
         padded_block_idxs, expert_offsets,
         gates, output_expanded) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out
        # print("backward")
        if gates is not None:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            grouped_grad_out = output_expanded.flatten(0, 1) # reuse expanded buffer later
        else:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            grouped_grad_out = ops.group(grad_out, sorted_scattered_idxs,
                                                 fan_out=gate_fan, coeff=gates_flat,
                                                 out=grouped_grad_out)
        if grouped_in:
            grouped_x = x
            d_expanded_input = None
        else:
            grouped_x = ops.group(x, sorted_scattered_idxs, fan_out=k)
            d_expanded_input = grouped_x
        d_weights = ops.group_bwd_W(
            DY=grouped_grad_out, X=grouped_x,
            expert_offsets=expert_offsets,
            E=expert_weights.size(0)
        )
        d_expanded_input = ops.scatter2scatter(
            X=grouped_grad_out, x_grouped=True,
            W=expert_weights.permute(0, 2, 1),
            padded_block_idxs=padded_block_idxs,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1,
            y_grouped=grouped_in,
            out=d_expanded_input # Reuse grouped_x buffer
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)
        # print("backward end.")
        return (
            # x, expert_weights, k,
            d_input, d_weights, None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None, None,
            # padded_block_idxs, expert_offsets,
            None, None,
            # gates
            d_gates, None, None
        )

def parallel_linear(inputs, expert_weights, k,
                    sorted_expert_idxs, sorted_scattered_idxs,
                    padded_block_idxs, expert_offsets,
                    gates=None):
    results = ParallelLinear.apply(inputs, expert_weights, k,
                                   sorted_expert_idxs, sorted_scattered_idxs,
                                   padded_block_idxs, expert_offsets, gates)
    return results

class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.num_experts, self.input_size, self.output_size)

    def forward(self, inputs, k, sorted_expert_idxs, sorted_scattered_idxs,
                padded_block_idxs, expert_offsets,
                gates=None, grouped_in=False, grouped_out=False):

        results = ParallelLinear.apply(
            inputs, self.weight.permute(0, 2, 1), k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates, grouped_in, grouped_out
        )
        return results