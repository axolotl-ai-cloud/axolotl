"""Tests for SiLU activation function Triton kernels."""
# pylint: disable=duplicate-code

import torch
import torch.nn.functional as F

from axolotl.kernels.silu import silu_backward, silu_forward


def test_silu_forward_shape():
    """Test that SiLU forward pass preserves expected shapes"""
    batch, seq_len, hidden_dim = 2, 3, 64
    x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    out = silu_forward(x)
    assert out.shape == (batch, seq_len, hidden_dim)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_silu_forward_values():
    """Test SiLU forward pass matches PyTorch reference implementation"""
    x = torch.randn(2, 3, 64, device="cuda")

    # Custom implementation
    triton_out = silu_forward(x.clone())

    # PyTorch reference
    torch_out = F.silu(x)

    assert torch.allclose(triton_out, torch_out, rtol=1e-3)


def test_silu_backward():
    """Test SiLU backward pass matches PyTorch autograd"""
    x = torch.randn(2, 3, 64, device="cuda", requires_grad=True)
    grad_output = torch.randn(2, 3, 64, device="cuda")

    # PyTorch reference - compute intermediates
    torch_out = F.silu(x)
    torch_out.backward(grad_output)

    # Custom backward pass
    x_clone = x.clone().detach()
    grad_output_clone = grad_output.clone()

    h, our_grad_x = silu_backward(grad_output_clone, x_clone)

    # Compare outputs and gradients
    assert torch.allclose(h, torch_out, rtol=1e-3)
    assert torch.allclose(our_grad_x, x.grad, rtol=1e-3)


def test_silu_inplace_preservation():
    """Test that SiLU backward doesn't modify original tensors unexpectedly"""
    x = torch.randn(2, 3, 64, device="cuda")
    grad_output = torch.randn(2, 3, 64, device="cuda")

    x_copy = x.clone()
    grad_copy = grad_output.clone()

    silu_backward(grad_output, x)

    assert not torch.equal(x, x_copy), "Input should be modified in-place"
    assert not torch.equal(
        grad_output, grad_copy
    ), "Grad output should be modified in-place"
