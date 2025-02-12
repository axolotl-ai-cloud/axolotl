"""
Module for definition of SiLU / Swish Triton kernels.

See "Swish: a Self-Gated Activation Function" (https://arxiv.org/pdf/1710.05941v1).
"""
# pylint: disable=invalid-name,unnecessary-lambda-assignment,duplicate-code

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiLU forward kernel. The kernel computes activation in fp32 precision for better
    numerical stability, then converts back to original dtype for the final result.

    Args:
        x_ptr: Pointer to input tensor `[*, hidden_dim]`.
        out_ptr: Pointer to output tensor `[*, hidden_dim]`.
        n_elements: Total number of elements in the input tensor.
        BLOCK_SIZE: Size of thread blocks for parallel computation.
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input in fp32 for better numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Compute activation in fp32 then convert back to original dtype
    result = x * tl.sigmoid(x)
    result = result.to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _silu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiLU backward kernel. Stores gradient results in-place.

    Args:
        grad_out_ptr: Pointer to gradient output tensor `[*, hidden_dim]`.
        x_ptr: Pointer to input tensor `[*, hidden_dim]`.
        n_elements: Total number of elements in the input tensor.
        BLOCK_SIZE: Size of thread blocks for parallel computation.

    Note:
        After kernel execution, tensors are modified in-place:
        - `grad_out_ptr` contains forward output (`h`)
        - `x_ptr` contains gradient w.r.t input (`grad_x`)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values in fp32 for better precision
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Compute SiLU and forward output
    sigmoid_x = tl.sigmoid(x)
    silu_x = sigmoid_x * x
    silu_x = silu_x.to(grad_out.dtype)

    # Compute gradient
    # d/dx(x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    grad_x = grad_out.to(tl.float32) * (sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x))
    grad_x = grad_x.to(grad_out.dtype)

    # Store results
    tl.store(grad_out_ptr + offsets, silu_x, mask=mask)  # forward output
    tl.store(x_ptr + offsets, grad_x, mask=mask)  # gradient


def silu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU forward pass. Computes SiLU activation: `x * sigmoid(x)`.

    Args:
        x: Input tensor of shape `[batch, seq_len, hidden_dim]`.

    Returns:
        Output tensor of shape `[batch, seq_len, hidden_dim]`.
    """
    batch, seq_len, hidden_dim = x.shape
    n_elements = x.numel()
    out = torch.empty((batch, seq_len, hidden_dim), dtype=x.dtype, device="cuda")

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _silu_fwd_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return out


def silu_backward(
    grad_output: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SiLU backward pass using in-place operations.

    Args:
        grad_output: Gradient of loss with respect to output, shape `[batch, seq_len, hidden_dim]`.
        x: Input tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.

    Returns:
        Tuple containing:
            - Forward pass output (`h`)
            - Gradient with respect to input (`grad_x`)
    """
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _silu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        x_ptr=x,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    # After kernel execution, tensors contain:
    # grad_output: h (forward output)
    # x: grad_x (gradient wrt input)
    return grad_output, x
