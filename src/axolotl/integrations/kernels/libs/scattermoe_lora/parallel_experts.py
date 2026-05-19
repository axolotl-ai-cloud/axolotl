# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shawntan/scattermoe
# Copyright (c) Shawn Tan and ScatterMoE Contributors
# Licensed under the Apache License, Version 2.0
# See https://github.com/shawntan/scattermoe/blob/main/LICENSE

from typing import Optional

import torch
import torch.nn as nn

from . import kernels

# When the maximum addressable element offset across any input/output buffer
# exceeds INT_MAX, the Triton kernel's int32 pointer-offset arithmetic
# overflows. ``_needs_int64_indices`` returns True iff any tensor has
# ``numel() >= INT_MAX``, which is a sufficient condition for the
# ``M_idx * stride_*m`` product to overflow somewhere in the kernel. When
# True, callers pass ``INT64_INDICES=True`` to the kernel so the index range
# is cast to int64 before it enters the multiplication. Strides themselves
# are already int64 at the Python level (from ``tensor.stride()``); only
# the *index* type needs the bump.
#
# The threshold here matches the kernel's correctness boundary: as soon as
# any indexed buffer has 2**31 - 1 or more elements, the int32 multiply at
# the end of the buffer can overflow. The wrapper used to chunk the call to
# keep ``rows * y_dim`` below 2**31; with INT64_INDICES the kernel itself
# handles overflow, so the auto-dispatch routes directly to a single
# kernel launch in either mode.
_INT_MAX = 2**31 - 1


def _needs_int64_indices(*tensors) -> bool:
    """True iff any input/output tensor's element count exceeds INT_MAX."""
    for t in tensors:
        if t is None or not isinstance(t, torch.Tensor):
            continue
        if t.numel() >= _INT_MAX:
            return True
    return False


@torch.library.custom_op("scattermoe::bincount", mutates_args={})
def compileable_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength)


@compileable_bincount.register_fake
def _(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, dtype=torch.long, device=x.device)


@torch.compile
def flatten_sort_count(expert_idxs: torch.Tensor, num_experts: int):
    with torch.no_grad():
        flattened_expert_idxs = expert_idxs.flatten()
        sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flattened_expert_idxs)
        expert_counts = compileable_bincount(
            flattened_expert_idxs, minlength=num_experts
        )
        expert_offsets = expert_counts.cumsum(-1)
        return sorted_expert_idxs, sorted_scattered_idxs, expert_offsets


class ParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_biases: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ):
        # Cast weights to match input dtype (e.g. 8-bit LoRA)
        if expert_weights.dtype != x.dtype:
            expert_weights = expert_weights.to(x.dtype)
        if expert_biases is not None and expert_biases.dtype != x.dtype:
            expert_biases = expert_biases.to(x.dtype)
        L_scattered = sorted_expert_idxs.size(0)
        y_dim = expert_weights.size(-1)
        # Cheap probe: the kernel's overflow risk is the M_block * stride_ym
        # product, dominated by the output buffer L_scattered * y_dim. We also
        # check x because the X_ptr arithmetic uses similar indices.
        needs_int64_fwd = (L_scattered * y_dim) >= _INT_MAX or _needs_int64_indices(x)
        with torch.device(x.device):
            output = kernels.ops.scatter2scatter(
                X=x,
                W=expert_weights,
                b=expert_biases,
                k=k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                x_grouped=grouped_in,
                y_grouped=grouped_out,
                int64_indices=needs_int64_fwd,
            )
            if gates is not None:
                output_expanded = output.view(
                    gates.size(0), gates.size(1), output.size(-1)
                )
                output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
            else:
                output_expanded = None

            ctx.save_for_backward(
                x,
                expert_weights,
                expert_biases,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                gates,
                output_expanded,
            )
            ctx.grouped_in = grouped_in
            ctx.grouped_out = grouped_out
            ctx.k = k
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        with torch.device(grad_out.device):
            (
                x,
                expert_weights,
                expert_biases,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                gates,
                output_expanded,
            ) = ctx.saved_tensors
            k = ctx.k
            grouped_in = ctx.grouped_in
            grouped_out = ctx.grouped_out

            if gates is not None:
                # calculate gates gradient
                # d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
                d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)
                grouped_grad_out = output_expanded.flatten(
                    0, 1
                )  # reuse expanded buffer later
            else:
                d_gates = None
                gates_flat = None
                gate_fan = 1
                grouped_grad_out = None

            if grouped_out:
                grouped_grad_out = grad_out
            else:
                grouped_grad_out = kernels.ops.group(
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
                grouped_x = kernels.ops.group(x, sorted_scattered_idxs, fan_out=k)
                d_expanded_input = grouped_x

            d_weights, d_biases = kernels.ops.group_bwd_W(
                DY=grouped_grad_out,
                X=grouped_x,
                expert_offsets=expert_offsets,
                E=expert_weights.size(0),
                has_bias=expert_biases is not None,
            )

            L_scattered = sorted_expert_idxs.size(0)
            dx_dim = expert_weights.size(1)  # K dim of W = output dim for dX
            needs_int64_bwd = (
                L_scattered * dx_dim
            ) >= _INT_MAX or _needs_int64_indices(grouped_grad_out)
            d_expanded_input = kernels.ops.scatter2scatter(
                X=grouped_grad_out,
                x_grouped=True,
                W=expert_weights.permute(0, 2, 1),
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=1,
                y_grouped=grouped_in,
                out=d_expanded_input,  # Reuse grouped_x buffer
                int64_indices=needs_int64_bwd,
            )

            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(
                    x.size(0), k, d_expanded_input.size(-1)
                ).sum(-2)
        return (
            # x, expert_weights,
            d_input,
            d_weights,
            # k, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            None,
            None,
            None,
            None,
            # bias, gates
            d_biases,
            d_gates,
            # grouped_in, grouped_out,
            None,
            None,
        )


def parallel_linear(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    expert_biases=None,
    gates=None,
    grouped_in=False,
    grouped_out=False,
):
    results = ParallelLinear.apply(
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
    return results


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_size))
        else:
            self.bias = None

        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.reset_parameters()

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(
            self.num_experts, self.input_size, self.output_size
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        results = parallel_linear(
            inputs,
            self.weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            expert_biases=self.bias,
            gates=gates,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )
        return results
