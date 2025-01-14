"""
Module for definition of SwiGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
"""
# pylint: disable=invalid-name,duplicate-code

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU forward kernel."""
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate in fp32, keep up in original dtype
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Compute in fp32 then convert back
    f = gate * tl.sigmoid(gate)
    f = f.to(up.dtype)
    result = f * up

    tl.store(out_ptr + offsets, result, mask=mask)


# pylint: disable=unnecessary-lambda-assignment
def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Direct wrapper for SwiGLU forward kernel."""
    batch, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hidden_dim), dtype=gate.dtype, device="cuda")

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _swiglu_fwd_kernel[grid](
        gate_ptr=gate,
        up_ptr=up,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return out


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
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    sigmoid_gate = tl.sigmoid(gate)
    f = gate * sigmoid_gate
    df = sigmoid_gate * (1 - f) + f

    f = f.to(up.dtype)
    df = df.to(up.dtype)

    grad_gate = grad_out * df * up
    grad_up = grad_out * f

    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up, mask=mask)


# pylint: disable=unnecessary-lambda-assignment
def swiglu_backward(grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """Direct wrapper for SwiGLU backward kernel."""
    n_elements = grad_out.numel()
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _swiglu_bwd_kernel[grid](
        grad_out_ptr=grad_out,
        gate_ptr=gate,
        up_ptr=up,
        grad_gate_ptr=grad_gate,
        grad_up_ptr=grad_up,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return grad_gate, grad_up
