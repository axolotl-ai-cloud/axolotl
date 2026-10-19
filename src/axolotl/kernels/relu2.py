"""ReLU-squared activation and backward fusion for non-gated dense MLPs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _relu2_forward(x, out, count, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < count, other=0)
    positive = tl.maximum(value, 0)
    tl.store(out + index, positive * positive, index < count)


@triton.jit
def _relu2_backward(grad, x, out, count, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < count, other=0)
    upstream = tl.load(grad + index, index < count, other=0)
    derivative = (2.0 * tl.maximum(value.to(tl.float32), 0.0)).to(value.dtype)
    tl.store(out + index, upstream * derivative, index < count)


def relu2_forward(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return x.clamp_min(0).square()
    x = x.contiguous()
    result = torch.empty_like(x)
    _relu2_forward[(triton.cdiv(x.numel(), 1024),)](x, result, x.numel(), BLOCK=1024)
    return result


def relu2_backward(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return grad * (2 * x.clamp_min(0))
    result = torch.empty_like(x)
    _relu2_backward[(triton.cdiv(x.numel(), 1024),)](
        grad.contiguous(), x.contiguous(), result, x.numel(), BLOCK=1024
    )
    return result


class _ReLU2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu2_forward(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return relu2_backward(grad, x)


def apply_relu2(x: torch.Tensor) -> torch.Tensor:
    return _ReLU2.apply(x)
