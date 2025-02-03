"""
Module for definition of SwiGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""
# pylint: disable=invalid-name,unnecessary-lambda-assignment

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

    # Compute activation in fp32 then convert back
    f = gate * tl.sigmoid(gate)
    f = f.to(up.dtype)
    result = f * up

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU backward kernel - storing results in-place following unsloth."""
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Forward pass computations needed for backward
    sigmoid_gate = tl.sigmoid(gate)
    f = sigmoid_gate * gate
    f = f.to(grad_out.dtype)
    h = f * up

    # Compute gradients
    df = grad_out * f  # grad wrt f
    dg = grad_out * up  # grad wrt up
    de = dg.to(tl.float32) * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    de = de.to(grad_out.dtype)

    # Store results in-place
    tl.store(grad_out_ptr + offsets, h, mask=mask)  # forward output
    tl.store(gate_ptr + offsets, df, mask=mask)  # grad wrt gate
    tl.store(up_ptr + offsets, de, mask=mask)  # grad wrt up


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU forward pass."""
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


def swiglu_backward(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """SwiGLU backward pass using in-place operations."""
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _swiglu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        gate_ptr=gate,
        up_ptr=up,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    # After kernel execution, tensors contain:
    # grad_output: h (forward output)
    # gate: df (grad wrt gate)
    # up: de (grad wrt up)
    return grad_output, gate, up
