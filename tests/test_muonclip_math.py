"""Tests for Muon math helpers."""

import torch

from axolotl.muonclip.math import muon_orthogonal_update, newton_schulz_orthogonalize


def test_newton_schulz_returns_orthogonal_matrix():
    tensor = torch.randn(4, 4)
    ortho = newton_schulz_orthogonalize(tensor, steps=4)
    prod = ortho @ ortho.transpose(-2, -1)
    off_diag = prod - torch.diag(torch.diagonal(prod))
    assert torch.allclose(
        torch.zeros_like(off_diag), off_diag, atol=0.3
    ), "Off-diagonal terms should stay small"


def test_muon_orthogonal_update_shapes_match():
    grad = torch.randn(2, 3, requires_grad=False)
    momentum = torch.zeros_like(grad)
    update = muon_orthogonal_update(
        grad,
        momentum,
        beta=0.95,
        ns_steps=2,
    )
    assert update.shape == grad.shape
