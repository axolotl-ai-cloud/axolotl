"""Tests for GELU activation function Triton kernels."""
# pylint: disable=duplicate-code

import torch
import torch.nn.functional as F

from axolotl.kernels.gelu import gelu_backward, gelu_forward


def test_gelu_forward_shape():
    """Test that GELU forward pass preserves expected shapes."""
    batch, seq_len, hidden_dim = 2, 3, 64
    x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    out = gelu_forward(x)
    assert out.shape == (batch, seq_len, hidden_dim)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_gelu_forward_values():
    """Test GELU forward pass matches PyTorch reference implementation."""
    x = torch.randn(2, 3, 64, device="cuda")

    # Custom implementation
    triton_out = gelu_forward(x.clone())

    # PyTorch reference
    torch_out = F.gelu(x)

    assert torch.allclose(triton_out, torch_out, rtol=1e-3)


def test_gelu_backward():
    """Test GELU backward pass matches PyTorch autograd."""
    x = torch.randn(2, 3, 64, device="cuda", requires_grad=True)
    grad_output = torch.randn(2, 3, 64, device="cuda")

    # PyTorch reference - compute intermediates
    torch_out = F.gelu(x)
    torch_out.backward(grad_output)

    # Custom backward pass
    x_clone = x.clone().detach()
    grad_output_clone = grad_output.clone()

    h, grad_x = gelu_backward(grad_output_clone, x_clone)

    # Compare outputs and gradients
    assert torch.allclose(h, torch_out, rtol=5e-3)
    assert torch.allclose(grad_x, x.grad, rtol=5e-3)


def test_gelu_inplace_preservation():
    """Test that GELU backward doesn't modify original tensors unexpectedly."""
    x = torch.randn(2, 3, 64, device="cuda")
    grad_output = torch.randn(2, 3, 64, device="cuda")

    x_copy = x.clone()
    grad_copy = grad_output.clone()

    gelu_backward(grad_output, x)

    assert not torch.equal(x, x_copy), "Input should be modified in-place"
    assert not torch.equal(
        grad_output, grad_copy
    ), "Grad output should be modified in-place"
