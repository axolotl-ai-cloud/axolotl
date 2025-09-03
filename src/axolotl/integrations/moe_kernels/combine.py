# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


# Adding out-of-tree ops to the `symm_mem` library
lib = torch.library.Library("symm_mem", "FRAGMENT")  # noqa: TOR901

"""
all_to_all_vdev_2d_copy:
Copy data from `input` to `symm_in_buf` and call `all_to_all_vdev_2d` to shuffle data
"""
lib.define(
    "all_to_all_vdev_2d_copy("
    "Tensor input, Tensor symm_in_buf, Tensor(a!) out, "
    "Tensor in_splits, Tensor(a!) out_splits_offsets, str group_name, int? major_align=None) -> ()",
)


@torch.library.impl(lib, "all_to_all_vdev_2d_copy", "CUDA")
def _all_to_all_vdev_2d_copy_cuda(
    input: torch.Tensor,
    symm_in_buf: torch.Tensor,
    out: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    major_align: Optional[int] = None,
) -> None:
    if symm_in_buf.shape[0] < input.shape[0]:
        raise RuntimeError(
            f"symm_in_buf with dim-0 length {symm_in_buf.shape[0]} cannot fit input with dim-0 length  {input.shape[0]}"
        )
    if symm_in_buf.shape[1:] != input.shape[1:]:
        raise RuntimeError(
            f"symm_in_buf non-0 dims do not match that of input: {symm_in_buf.shape[1:]} vs {input.shape[1:]}"
        )
    if symm_in_buf.dtype != input.dtype:
        raise RuntimeError(
            f"symm_in_buf dtype {symm_in_buf.dtype} does not match input dtype {input.dtype}"
        )

    symm_in_buf.narrow(0, 0, input.shape[0]).copy_(input)
    torch.ops.symm_mem.all_to_all_vdev_2d(
        symm_in_buf,
        out,
        in_splits,
        out_splits_offsets,
        group_name,
        major_align,
    )


class AllToAllVDev2dOffset(torch.autograd.Function):
    """
    Autograd function for `all_to_all_vdev_2d_offset`
    """

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        input: torch.Tensor,
        out: torch.Tensor,
        in_splits_offsets: torch.Tensor,
        out_splits_offsets: torch.Tensor,
        group_name: str,
        # Buffers needed for backward pass
        major_align: int,
        grad_out_buf: torch.Tensor,
        grad_in_buf: torch.Tensor,
        grad_in_splits_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Functionality is the same as `all_to_all_vdev_2d_offset` but with functionalization.
        """
        # Shuffle input to output
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            input, out, in_splits_offsets, out_splits_offsets, group_name
        )

        # Output splits in forward is the input splits in backward
        ctx.save_for_backward(
            out_splits_offsets, grad_out_buf, grad_in_buf, grad_in_splits_offsets
        )
        ctx.group_name = group_name
        ctx.major_align = major_align
        return out

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None]:
        """
        Backward pass of `all_to_all_vdev_2d_offset` is `all_to_all_vdev_2d`.

        Args:
            `grad_output`: gradients of output passed back from the downstream.

        Returns:
            `grad_input`: gradients of input.
        """
        # Splits info
        # Splits/offsets of grad_out is the same as out splits/offsets in forward
        (
            grad_out_splits_offsets,
            grad_out_buf,
            grad_in_buf,
            grad_in_splits_offsets,
        ) = ctx.saved_tensors
        grad_out_splits = grad_out_splits_offsets[0]

        # Shuffle gradients back to the input
        # TODO: create an op that takes both in_splits_offsets and
        # out_splits_offsets, instead of taking alignment
        torch.ops.symm_mem.all_to_all_vdev_2d_copy(
            grad_output,
            grad_out_buf,
            grad_in_buf,
            grad_out_splits,
            grad_in_splits_offsets,
            ctx.group_name,
            ctx.major_align,
        )

        return grad_in_buf, None, None, None, None, None, None, None, None


class TokenCombiner(torch.nn.Module):
    """
    Combine tokens from different experts, with backward pass to shuffle gradients back to the input.
    Args:
        `group_name`: name of the group to use for communication.
        `align`: alignment of the token offsets from each expert. If using
                 Grouped Gemm next, this should be the same as Grouped Gemm's alignment.
        `in_len`: length of the input.
        `out_len`: length of the output.
        `token_shape`: shape of the tokens.
        `num_ranks`: number of ranks in the group.
        `num_local_experts`: number of local experts.
        `dtype`: data type of the input/output.
    """

    def __init__(
        self,
        group_name: str,
        align: int,
        in_len,
        out_len,
        token_shape,
        num_ranks,
        num_local_experts,
        dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.group_name = group_name
        self.align = align
        self.grad_out_buf = symm_mem.empty(
            out_len, *token_shape, dtype=dtype, device=device
        )
        self.grad_in_buf = symm_mem.empty(
            in_len, *token_shape, dtype=dtype, device=device
        )
        self.nsplits = num_ranks * num_local_experts
        self.grad_in_splits_offsets = symm_mem.empty(
            (2, self.nsplits), dtype=torch.int64, device=device
        )

    def forward(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        in_splits_offsets: torch.Tensor,
        out_splits_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `inp`: input tensor.
            `out`: buffer for output tensor.
            `in_splits_offsets`: splits and offsets of the input tensor.
            `out_splits_offsets`: splits and offsets of the output tensor.
            See `all_to_all_vdev_2d_offset` for more details.
        Note:
            All tensor arguments must be symmetrically allocated, i.e.
            >>> inp = symm_mem.empty(max_inp_len, dtype=dtype, device=device)
            >>> out = symm_mem.empty(max_out_len, dtype=dtype, device=device)
            >>> in_splits_offsets = symm_mem.empty(
            ...     (2, nsplits), dtype=torch.int64, device=device)
            >>> out_splits_offsets = symm_mem.empty(
            ...     (2, nsplits), dtype=torch.int64, device=device)
        """

        if in_splits_offsets.shape != (2, self.nsplits):
            raise ValueError(
                f"Expected shape (2, {self.nsplits}), got {in_splits_offsets.shape}"
            )
        if out_splits_offsets.shape != (2, self.nsplits):
            raise ValueError(
                f"Expected shape (2, {self.nsplits}), got {out_splits_offsets.shape}"
            )

        return AllToAllVDev2dOffset.apply(
            inp,
            out,
            in_splits_offsets,
            out_splits_offsets,
            self.group_name,
            self.align,
            self.grad_out_buf,
            self.grad_in_buf,
            self.grad_in_splits_offsets,
        )