"""
Module for definition of SwiGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    """
    SwiGLU forward kernel. The kernel computes activation in fp32 precision for better
    numerical stability, then converts back to original dtype for the final result.

    Args:
        gate_ptr: Pointer to gate tensor `[*, hidden_dim]`.
        up_ptr: Pointer to up-projection tensor `[*, hidden_dim]`.
        out_ptr: Pointer to output tensor `[*, hidden_dim]`.
        n_elements: Total number of elements in the input tensors.
        block_size: Size of thread blocks for parallel computation.
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * block_size + tl.arange(0, block_size)
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
    block_size: tl.constexpr,
):
    """
    SwiGLU backward kernel. Stores gradient results in-place.

    Args:
        grad_out_ptr: Pointer to gradient output tensor `[*, hidden_dim]`.
        gate_ptr: Pointer to gate tensor `[*, hidden_dim]`.
        up_ptr: Pointer to up-projection tensor `[*, hidden_dim]`.
        n_elements: Total number of elements in the input tensors.
        block_size: Size of thread blocks for parallel computation.

    Note:
        After kernel execution, tensors are modified in-place:
        - `grad_out_ptr` contains forward output (`h`)
        - `gate_ptr` contains gradient w.r.t gate (`grad_gate`)
        - `up_ptr` contains gradient w.r.t up (`grad_up`)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load values - only convert gate to fp32
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Compute SiLU and forward output
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = sigmoid_gate * gate
    silu_gate = silu_gate.to(grad_out.dtype)
    h = silu_gate * up

    # Compute gradients
    grad_up = grad_out * silu_gate  # gradient for up is grad_out * SiLU(gate)

    # Compute gate gradient
    temp = grad_out * up
    grad_gate = temp.to(tl.float32) * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_gate = grad_gate.to(grad_out.dtype)

    # Store results with correct gradient ordering
    tl.store(grad_out_ptr + offsets, h, mask=mask)
    tl.store(gate_ptr + offsets, grad_gate, mask=mask)  # grad wrt gate
    tl.store(up_ptr + offsets, grad_up, mask=mask)  # grad wrt up


# pylint: disable=unnecessary-lambda-assignment
def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU forward pass. Computes SwiGLU activation: `x * sigmoid(x) * up`, where
    `x` is the gate tensor.

    Args:
        gate: Input gate tensor of shape `[batch, seq_len, hidden_dim]`.
        up: Up-projection tensor of shape `[batch, seq_len, hidden_dim]`.

    Returns:
        Output tensor of shape `[batch, seq_len, hidden_dim]`.
    """
    batch, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hidden_dim), dtype=gate.dtype, device="cuda")

    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)  # noqa: E731
    _swiglu_fwd_kernel[grid](
        gate_ptr=gate,
        up_ptr=up,
        out_ptr=out,
        n_elements=n_elements,
        block_size=1024,
    )

    return out


# pylint: disable=unnecessary-lambda-assignment
def swiglu_backward(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SwiGLU backward pass using in-place operations.

    Args:
        grad_output: Gradient of loss with respect to output, shape `[batch, seq_len, hidden_dim]`.
        gate: Gate tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.
        up: Up-projection tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.

    Returns:
        Tuple containing:
            - Forward pass output (`h`)
            - Gradient with respect to gate (`df`)
            - Gradient with respect to up-projection (`de`)
    """
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)  # noqa: E731
    _swiglu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        gate_ptr=gate,
        up_ptr=up,
        n_elements=n_elements,
        block_size=1024,
    )

    # After kernel execution, tensors contain:
    # grad_output: h (forward output)
    # gate: grad_gate (grad wrt gate)
    # up: grad_up (grad wrt up)
    return grad_output, gate, up
