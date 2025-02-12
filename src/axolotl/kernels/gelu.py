"""
Module for definition of GELU Triton kernels.

See "Gaussian Error Linear Units (GELUs)" (https://arxiv.org/abs/1606.08415).
"""
# pylint: disable=invalid-name,unnecessary-lambda-assignment,duplicate-code

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU forward kernel.

    Args:
        x_ptr: Pointer to input tensor [*, hidden_dim].
        out_ptr: Pointer to output tensor [*, hidden_dim].
        n_elements: Total number of elements in the input tensor.
        BLOCK_SIZE: Size of thread blocks for parallel computation.
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Compute activation in fp32 then convert back
    # GELU(x) = 0.5x * (1 + erf(x/sqrt(2)))
    gelu = 0.5 * x * (tl.math.erf(tl.math.rsqrt(2.0) * x) + 1.0)
    gelu = gelu.to(x.dtype)

    tl.store(out_ptr + offsets, gelu, mask=mask)


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """GELU forward pass.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim].

    Returns:
        torch.Tensor: Output tensor of shape [batch, seq_len, hidden_dim].
    """
    batch, seq_len, hidden_dim = x.shape
    n_elements = x.numel()
    out = torch.empty((batch, seq_len, hidden_dim), dtype=x.dtype, device="cuda")

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _gelu_fwd_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    return out


@triton.jit
def _gelu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU backward kernel. Stores gradient results in-place.

    Args:
        grad_out_ptr: Pointer to gradient output tensor [*, hidden_dim].
        x_ptr: Pointer to input tensor [*, hidden_dim].
        n_elements: Total number of elements in the input tensors.
        BLOCK_SIZE: Size of thread blocks for parallel computation.

    Note:
        After kernel execution, tensors are modified in-place:
        - `grad_out_ptr` contains GELU activation output (`h`)
        - `x_ptr` contains gradient w.r.t input (`grad_x`)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Forward pass
    gelu_partial = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * x) + 1.0)
    gelu = gelu_partial * x

    # Compute gradient
    # d/dx(GELU(x)) = 0.5 * (erf(x/sqrt(2)) + 1) + x * exp(-x^2/2) / sqrt(2*pi)
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    dgelu_dx = gelu_partial + t * x * tl.exp(-0.5 * x * x)
    grad_x = grad_out.to(tl.float32) * dgelu_dx

    # Convert outputs back to original dtype
    gelu = gelu.to(grad_out.dtype)
    grad_x = grad_x.to(grad_out.dtype)

    # Store results
    tl.store(grad_out_ptr + offsets, gelu, mask=mask)
    tl.store(x_ptr + offsets, grad_x, mask=mask)


def gelu_backward(
    grad_output: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """GELU backward pass using in-place operations.

    Args:
        grad_output: Gradient of loss with respect to output, shape `[batch, seq_len, hidden_dim]`.
        x: Input tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.

    Returns:
        Tuple containing:
            - GELU activation output (`h`)
            - Gradient with respect to input (`grad_x`)

    Note:
        This function modifies its input tensors in-place to store results.
    """
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _gelu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        x_ptr=x,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return grad_output, x
