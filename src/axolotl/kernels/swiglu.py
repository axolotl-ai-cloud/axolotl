"""
Module for definition of SwiGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
"""
# pylint: disable=invalid-name,duplicate-code

import torch
import triton
import triton.language as tl

from axolotl.kernels.utils import calculate_grid


@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU forward kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs in fp32 for numerical stability
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Compute activation in fp32
    result = gate * tl.sigmoid(gate) * up

    # Convert back to original dtype for storage
    result = result.to(up.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU backward kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load all inputs in fp32 for stability
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Compute gradients in fp32
    sigmoid_gate = tl.sigmoid(gate)
    f = gate * sigmoid_gate
    df = sigmoid_gate * (1 - f) + f

    grad_gate = grad_out * df * up
    grad_up = grad_out * f

    # Convert back to original dtype for storage
    out_dtype = grad_out.dtype
    tl.store(grad_gate_ptr + offsets, grad_gate.to(out_dtype), mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up.to(out_dtype), mask=mask)


class SwiGLU:
    """
    Swish Gated Linear Unit (SwiGLU) implementation using Triton kernels for forward
    and backward passes.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU activation."""
        n_elements = gate.numel()
        out = torch.empty_like(gate)

        grid, block_size, num_warps = calculate_grid(n_elements)

        _swiglu_fwd_kernel[grid](
            gate_ptr=gate,
            up_ptr=up,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
        """Backward pass for SwiGLU activation."""
        n_elements = grad_out.numel()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)

        grid, block_size, num_warps = calculate_grid(n_elements)

        _swiglu_bwd_kernel[grid](
            grad_out_ptr=grad_out,
            gate_ptr=gate,
            up_ptr=up,
            grad_gate_ptr=grad_gate,
            grad_up_ptr=grad_up,
            n_elements=n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        return grad_gate, grad_up
