# pylint: skip-file

# MIT License
#
# Copyright (c) 2024 mgmalek
# https://github.com/mgmalek/efficient_cross_entropy/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import sqrt

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_cross_entropy_fwd_bwd_kernel(
    output_loss_ptr,
    output_logit_grad_ptr,
    input_logit_ptr,
    input_targ_ptr,
    input_divisor_ptr,
    output_loss_stride,
    output_logit_grad_stride,
    input_logit_stride,
    input_targ_stride,
    n_cols,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    # Get pointers to current row for all inputs/outputs
    row_idx = tl.program_id(0)
    logit_grad_row_start_ptr = (
        output_logit_grad_ptr + row_idx * output_logit_grad_stride
    )
    logit_row_start_ptr = input_logit_ptr + row_idx * input_logit_stride
    targ_ptr = input_targ_ptr + row_idx * input_targ_stride
    loss_ptr = output_loss_ptr + row_idx * output_loss_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    logit_row_ptrs = logit_row_start_ptr + col_offsets
    logit_grad_row_ptrs = logit_grad_row_start_ptr + col_offsets

    # Load data into SRAM
    logit_row_unnormalized = tl.load(
        logit_row_ptrs, mask=col_offsets < n_cols, other=float("-Inf")
    )
    targ = tl.load(targ_ptr)
    divisor = tl.load(input_divisor_ptr)

    # Normalize logits and compute some useful intermediate values
    logit_row = logit_row_unnormalized - tl.max(
        logit_row_unnormalized, axis=0
    )  # Subtract max value for numerical stability
    exp_logit_row = tl.exp(logit_row)
    sum_exp_logit_row = tl.sum(exp_logit_row, axis=0)

    # Compute loss
    log_sum_exp_logit_row = tl.log(sum_exp_logit_row)
    logit_gt_logit = tl.sum(tl.where(targ == col_offsets, logit_row, 0.0))
    loss = log_sum_exp_logit_row - logit_gt_logit
    loss = loss / divisor
    loss = tl.where(targ == ignore_index, 0.0, loss)
    tl.store(loss_ptr, loss)

    # Compute gradients
    targ_one_hot = tl.where(targ == col_offsets, 1.0, 0.0)
    grad = exp_logit_row / sum_exp_logit_row - targ_one_hot
    grad = grad / divisor
    grad = tl.where(targ == ignore_index, 0.0, grad)
    tl.store(logit_grad_row_ptrs, grad, mask=col_offsets < n_cols)


class FusedCrossEntropyLossFunction(torch.autograd.Function):
    # NOTE: We put the linear projection in the same autograd Function as the loss computation
    # because we overwrite the logits with their gradients inplace to avoid allocating more
    # memory for the gradients, and so we keep the logits completely contained within this
    # Functionto avoid possible side-effects if they were exposed.

    @staticmethod
    def forward(
        ctx,
        in_feat: torch.Tensor,
        proj_weight: torch.Tensor,
        targ: torch.Tensor,
        n_loop_iters: int,
        ignore_index: int,
        reduction: str,
    ):
        n_tokens = in_feat.shape[0]
        n_classes = proj_weight.shape[0]

        assert in_feat.ndim == 2, in_feat.ndim
        assert proj_weight.ndim == 2, proj_weight.ndim
        assert targ.ndim == 1, targ.shape
        assert (
            in_feat.shape[0] == targ.shape[0]
        ), f"Number of tokens in in_feat and targ is not equal: {(in_feat.shape, targ.shape) = }"
        assert reduction in ("mean", "sum"), reduction
        assert n_loop_iters > 0, n_loop_iters
        assert n_tokens % n_loop_iters == 0, (n_tokens, n_loop_iters)

        NUM_WARPS = 16

        BLOCK_SIZE = triton.next_power_of_2(n_classes)

        loss = torch.empty(n_tokens, dtype=in_feat.dtype, device=in_feat.device)
        dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else in_feat.dtype
        )

        if proj_weight.requires_grad:
            grad_proj_weight = torch.zeros_like(proj_weight, dtype=dtype)
        else:
            grad_proj_weight = None

        if in_feat.requires_grad:
            grad_in_feat = torch.zeros_like(in_feat)
        else:
            grad_in_feat = None

        divisor = (
            (targ != ignore_index).sum().to(dtype)
            if reduction == "mean"
            else torch.ones(1, dtype=dtype, device=in_feat.device)
        )

        # Divide the input into chunks of size num_tokens // n_loop_iters, then compute the loss for each of these groups
        proj_weight_cast = proj_weight.to(dtype)

        loop_chunk_size = triton.cdiv(n_tokens, n_loop_iters)
        logits_chunk_cast = torch.zeros(
            (loop_chunk_size, n_classes), dtype=dtype, device=in_feat.device
        )
        for i, in_feat_chunk in enumerate(torch.split(in_feat, loop_chunk_size)):
            token_start_idx = i * loop_chunk_size
            token_end_idx = (i + 1) * loop_chunk_size

            in_feat_chunk = in_feat_chunk.to(dtype)

            # Compute logits
            torch.matmul(in_feat_chunk, proj_weight_cast.T, out=logits_chunk_cast)
            logits_chunk = logits_chunk_cast.float()

            # Compute loss
            loss_chunk = loss[token_start_idx:token_end_idx]
            targ_chunk = torch.zeros(
                loop_chunk_size, dtype=targ.dtype, device=targ.device
            )
            targ_chunk[: loop_chunk_size - 1] = targ[
                token_start_idx + 1 : token_end_idx
            ]
            if i == n_loop_iters - 1:
                targ_chunk[-1] = ignore_index
            else:
                targ_chunk[-1] = targ[token_end_idx + 1]

            n_tokens_chunk = logits_chunk.shape[0]
            grad_logits_chunk = (
                logits_chunk  # NOTE: we override the logits with their gradients
            )
            fused_cross_entropy_fwd_bwd_kernel[(n_tokens_chunk,)](
                loss_chunk,
                grad_logits_chunk,
                logits_chunk,
                targ_chunk,
                divisor,
                loss_chunk.stride(0),
                grad_logits_chunk.stride(0),
                logits_chunk.stride(0),
                targ_chunk.stride(0),
                n_classes,
                ignore_index,
                num_warps=NUM_WARPS,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            grad_logits_chunk = grad_logits_chunk.to(dtype)

            if in_feat.requires_grad:
                grad_in_feat[token_start_idx:token_end_idx] = (
                    grad_logits_chunk @ proj_weight_cast
                )

            if proj_weight.requires_grad:
                torch.addmm(
                    grad_proj_weight,
                    grad_logits_chunk.T,
                    in_feat_chunk,
                    out=grad_proj_weight,
                )

        # NOTE: if reduction == "mean" we already divide by an appropriate normalization factor in the kernel so we can alway sum here
        loss = loss.sum()

        # Save data for backward
        ctx.in_feat_requires_grad = in_feat.requires_grad
        ctx.proj_weight_requires_grad = proj_weight.requires_grad

        if proj_weight.requires_grad and in_feat.requires_grad:
            ctx.save_for_backward(grad_in_feat, grad_proj_weight)
        elif proj_weight.requires_grad and not in_feat.requires_grad:
            ctx.save_for_backward(grad_proj_weight)
        elif not proj_weight.requires_grad and in_feat.requires_grad:
            ctx.save_for_backward(grad_in_feat)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_in_feat = None
        grad_proj_weight = None
        if ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            grad_in_feat, grad_proj_weight = ctx.saved_tensors
        elif not ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            (grad_proj_weight,) = ctx.saved_tensors
        elif ctx.in_feat_requires_grad and not ctx.proj_weight_requires_grad:
            (grad_in_feat,) = ctx.saved_tensors

        assert grad_output.shape == tuple(), grad_output.shape
        if ctx.in_feat_requires_grad:
            grad_in_feat *= grad_output
        if ctx.proj_weight_requires_grad:
            grad_proj_weight *= grad_output

        return grad_in_feat, grad_proj_weight, None, None, None, None


class FusedProjectionPlusCrossEntropyLoss(nn.Module):
    """Fused implementation of linear projection + cross entropy loss"""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        n_loop_iters: int = 1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.n_loop_iters = n_loop_iters
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.proj_weight = nn.Parameter(torch.empty(n_classes, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.proj_weight, a=sqrt(5))

    def forward(self, x, targ):
        return FusedCrossEntropyLossFunction.apply(
            x,
            self.proj_weight,
            targ,
            self.n_loop_iters,
            self.ignore_index,
            self.reduction,
        )


class PyTorchProjectionPlusCrossEntropyLoss(nn.Module):
    """Simple PyTorch implementation of linear projection + cross entropy loss. Intended only for testing and benchmarking."""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.proj = nn.Linear(dim, n_classes, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, targ):
        logits = self.proj(x)
        return self.loss(logits, targ)
