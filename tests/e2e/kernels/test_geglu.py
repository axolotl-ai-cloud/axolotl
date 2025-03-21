"""Tests for GEGLU activation function Triton kernels."""

# pylint: disable=duplicate-code

import torch
import torch.nn.functional as F

from axolotl.kernels.geglu import geglu_backward, geglu_forward


def test_geglu_forward_shape():
    """Test that GEGLU forward pass preserves expected shapes."""
    batch, seq_len, hidden_dim = 2, 3, 64
    gate = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    up = torch.randn(batch, seq_len, hidden_dim, device="cuda")

    out = geglu_forward(gate, up)
    assert out.shape == (batch, seq_len, hidden_dim)
    assert out.dtype == gate.dtype
    assert out.device == gate.device


def test_geglu_forward_values():
    """Test GEGLU forward pass matches PyTorch reference implementation."""
    gate = torch.randn(2, 3, 64, device="cuda")
    up = torch.randn(2, 3, 64, device="cuda")

    # Custom implementation
    triton_out = geglu_forward(gate.clone(), up.clone())

    # PyTorch reference
    torch_out = F.gelu(gate) * up

    assert torch.allclose(triton_out, torch_out, rtol=1e-3)


def test_geglu_backward():
    """Test GEGLU backward pass matches PyTorch autograd."""
    gate = torch.randn(2, 3, 64, device="cuda", requires_grad=True)
    up = torch.randn(2, 3, 64, device="cuda", requires_grad=True)
    grad_output = torch.randn(2, 3, 64, device="cuda")

    # PyTorch reference - compute intermediates
    gelu_gate = F.gelu(gate)
    torch_out = gelu_gate * up
    torch_out.backward(grad_output)

    # Custom backward pass
    gate_clone = gate.clone().detach()
    up_clone = up.clone().detach()
    grad_output_clone = grad_output.clone()

    h, grad_gate, grad_up = geglu_backward(grad_output_clone, gate_clone, up_clone)

    # Compare outputs and gradients
    assert torch.allclose(h, torch_out, rtol=1e-3)
    assert torch.allclose(grad_gate, gate.grad, rtol=1e-3)
    assert torch.allclose(grad_up, up.grad, rtol=1e-3)


def test_geglu_inplace_preservation():
    """Test that GEGLU backward doesn't modify original tensors unexpectedly."""
    gate = torch.randn(2, 3, 64, device="cuda")
    up = torch.randn(2, 3, 64, device="cuda")
    grad_output = torch.randn(2, 3, 64, device="cuda")

    gate_copy = gate.clone()
    up_copy = up.clone()
    grad_copy = grad_output.clone()

    geglu_backward(grad_output, gate, up)

    assert not torch.equal(gate, gate_copy), "Gate should be modified in-place"
    assert not torch.equal(up, up_copy), "Up should be modified in-place"
    assert not torch.equal(
        grad_output, grad_copy
    ), "Grad output should be modified in-place"
