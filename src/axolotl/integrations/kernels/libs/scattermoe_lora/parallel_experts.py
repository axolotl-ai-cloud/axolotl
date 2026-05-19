# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shawntan/scattermoe
# Copyright (c) Shawn Tan and ScatterMoE Contributors
# Licensed under the Apache License, Version 2.0
# See https://github.com/shawntan/scattermoe/blob/main/LICENSE

from typing import Optional

import torch
import torch.nn as nn

from . import kernels
from .kernels.ops import BLOCK_M as _SCATTER2SCATTER_BLOCK_M

# Int32-overflow guard for the scatter2scatter Triton kernel.
#
# The kernel computes output-pointer offsets as
#   Y_ptr + M_block * stride_ym + N_block * stride_yn
# where M_block / N_block / stride_ym are int32. When the output buffer
# has L_scattered * y_dim >= 2**31 elements, the M_block * stride_ym
# product overflows int32 for the trailing rows. The masked stores then
# either silently drop (rows come back as zeros) or write to bogus
# addresses, which can also trip a delayed CUDA illegal-memory-access
# / CUBLAS_STATUS_EXECUTION_FAILED in a downstream kernel.
#
# This is reachable in production at seq=512K with coarse shards
# (tokens_per_shard * top_k * 2*INTERMEDIATE >= 2**31 elements). The
# fix is to chunk the call along the L_scattered axis when the kernel
# is in the only mode where chunking is safe — y_grouped=True, where
# the kernel uses M_block (a 0-based per-launch program_id) as the
# output row index. With a sliced sei/ssi and out=output[start:end],
# each sub-call's M_block * stride_ym fits in int32.
#
# Threshold is conservative (< 2**31 elements per call, not bytes) so
# the safety margin holds for fp32 / bf16 / fp16 alike. The check is
# guarded behind `if L_scattered * y_dim >= _SCATTER2SCATTER_INT32_LIMIT`
# so the common path (anything below ~2 GiB of bf16 output) goes
# straight to a single kernel launch with no extra overhead.
_SCATTER2SCATTER_INT32_LIMIT = 2**31
# Triton kernel's BLOCK_M tile is imported above as
# ``_SCATTER2SCATTER_BLOCK_M`` so the chunk boundaries stay in sync if
# ``kernels/ops.py`` ever retunes BLOCK_M.


def _scatter2scatter_int32_safe(
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    k: int,
    b: Optional[torch.Tensor] = None,
    x_grouped: bool = False,
    y_grouped: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrapper around ``kernels.ops.scatter2scatter`` that splits the call
    when its output would overflow the Triton kernel's int32 pointer
    arithmetic.

    Fast path (common case): if the output is below
    ``_SCATTER2SCATTER_INT32_LIMIT`` elements, dispatch one kernel call.
    Slow path (long-seq + coarse shards): for ``y_grouped=True``, allocate
    the full output and call the kernel on chunks of the sorted-index
    arrays, each writing to a sub-view of the output. ``y_grouped=False``
    is left unchunked because the kernel addresses output rows via the
    per-position scattered index ``M_idx`` (not a 0-based ``M_block``),
    so the row range can't be tiled safely from the wrapper; the only
    overflow-affected production call sites are ``y_grouped=True``.
    """
    L_scattered = sorted_expert_idxs.size(0)
    y_dim = W.size(-1)

    if L_scattered * y_dim < _SCATTER2SCATTER_INT32_LIMIT:
        return kernels.ops.scatter2scatter(
            X=X,
            W=W,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=k,
            b=b,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
            out=out,
        )

    if not y_grouped:
        # ``y_grouped=False`` uses per-position scattered indices for
        # the output rows, so the wrapper can't tile the row range
        # safely from the outside. Production overflow paths today
        # are all ``y_grouped=True`` so this branch is unreachable in
        # the bench; hard-raise here rather than fall through —
        # silently letting the raw kernel run would corrupt the
        # trailing rows. The kernel itself needs an int64
        # pointer-arithmetic fix before y_grouped=False at this scale
        # is safe.
        raise RuntimeError(
            f"scatter2scatter call with y_grouped=False has output "
            f"L_scattered ({L_scattered}) * y_dim ({y_dim}) = "
            f"{L_scattered * y_dim} elements >= the int32 overflow "
            f"threshold ({_SCATTER2SCATTER_INT32_LIMIT}). The Triton "
            f"kernel's pointer arithmetic would silently corrupt "
            f"output rows whose M_idx * stride_ym overflows int32. "
            f"The int32-safe wrapper cannot tile this case (it would "
            f"require per-position output-row tiling); the kernel "
            f"itself needs an int64 pointer-arithmetic fix."
        )

    if out is None:
        out = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
    else:
        # Mirror scatter2scatter's contract on out.
        assert out.size(0) == L_scattered and out.size(1) == y_dim

    # Chunk size — largest BLOCK_M-aligned row count whose
    # ``rows * y_dim`` product fits int32. The aligned floor cannot
    # be zero because y_dim must itself be below the int32 limit for
    # a single output row to fit (asserted below).
    max_rows = _SCATTER2SCATTER_INT32_LIMIT // y_dim
    chunk_rows = (max_rows // _SCATTER2SCATTER_BLOCK_M) * _SCATTER2SCATTER_BLOCK_M
    assert chunk_rows > 0, (
        f"scatter2scatter int32-safe chunking: y_dim={y_dim} is too large "
        f"to fit even one BLOCK_M-aligned chunk under the int32 limit"
    )
    # OOB-guard for x_grouped=False with a non-aligned last chunk: the
    # kernel's M_boundary_mask is ``M_block < FAN_OUT * X.size(0)``,
    # which equals the FULL L_scattered (X is not chunked in this
    # mode). A partial last chunk would let the kernel's final tile
    # tl.load past the end of sei_chunk / ssi_chunk and tl.store past
    # the end of out_chunk. The x_grouped=True path is naturally
    # bounded because X is chunked too (mask becomes
    # ``M_block < FAN_OUT * chunk_size``).
    #
    # For all realistic production callers (power-of-2 token counts
    # ✕ power-of-2 top_k, y_dim a clean power-of-2) ``L_scattered``
    # is a multiple of ``chunk_rows`` and the assertion holds. If a
    # future caller hits a non-aligned shape this assertion fires
    # rather than silently corrupting — the right fix at that point
    # is to teach the kernel to take a per-launch M override.
    if not x_grouped:
        assert L_scattered % chunk_rows == 0, (
            f"scatter2scatter int32-safe chunking with x_grouped=False "
            f"requires L_scattered ({L_scattered}) to be a multiple of "
            f"chunk_rows ({chunk_rows}); a partial last chunk would "
            f"trigger an out-of-bounds tl.load against sei_chunk because "
            f"the kernel's M_boundary_mask bounds against the full X "
            f"(unchunked) size, not the chunk size"
        )

    # When the X buffer is itself grouped (x_grouped=True), the kernel
    # uses the per-launch program_id (M_block) as the X row index, so
    # we must slice X in lockstep with the sei/ssi chunk so the kernel
    # reads X[chunk_start + M_block]. When x_grouped=False the kernel
    # indexes X via ``M_idx // FAN_OUT`` where M_idx is a global
    # (un-chunked) position from sorted_scattered_idxs, so X stays
    # full.
    #
    # The high-level scatter2scatter() wrapper asserts
    # ``sorted_scattered_idxs.size(0) == X.size(0) * k`` — true for the
    # full call but not for our chunks — so we drop into
    # ``scatter2scatter_compileable`` directly (the registered Triton
    # custom op, which carries no such precondition).
    for chunk_start in range(0, L_scattered, chunk_rows):
        chunk_end = min(chunk_start + chunk_rows, L_scattered)
        sei_chunk = sorted_expert_idxs[chunk_start:chunk_end]
        ssi_chunk = sorted_scattered_idxs[chunk_start:chunk_end]
        out_chunk = out[chunk_start:chunk_end]
        if x_grouped:
            X_chunk = X[chunk_start:chunk_end]
        else:
            X_chunk = X
        kernels.ops.scatter2scatter_compileable(
            out_chunk,
            W,
            X_chunk,
            k,
            sei_chunk,
            ssi_chunk,
            b,
            x_grouped,
            y_grouped,
        )
    return out


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
        with torch.device(x.device):
            output = _scatter2scatter_int32_safe(
                X=x,
                W=expert_weights,
                b=expert_biases,
                k=k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                x_grouped=grouped_in,
                y_grouped=grouped_out,
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

            d_expanded_input = _scatter2scatter_int32_safe(
                X=grouped_grad_out,
                x_grouped=True,
                W=expert_weights.permute(0, 2, 1),
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=1,
                y_grouped=grouped_in,
                out=d_expanded_input,  # Reuse grouped_x buffer
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
