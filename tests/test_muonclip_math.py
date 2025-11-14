"""Tests for Muon math helpers."""

import math

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


def _reference_muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    momentum_ref = momentum.clone()
    momentum_ref.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum_ref, beta)
    update_2d = update.view(update.shape[0], -1)
    update_2d = newton_schulz_orthogonalize(update_2d, steps=ns_steps)
    scale = math.sqrt(max(update_2d.size(0), update_2d.size(1)) * 0.4)
    update_2d = update_2d * scale
    return update_2d.view_as(grad), momentum_ref


def test_muon_update_matches_reference():
    torch.manual_seed(0)
    grad = torch.randn(4, 6)
    beta = 0.95
    ns_steps = 5

    reference_update, reference_momentum = _reference_muon_update(
        grad, torch.zeros_like(grad), beta=beta, ns_steps=ns_steps
    )
    momentum = torch.zeros_like(grad)
    update = muon_orthogonal_update(
        grad.clone(),
        momentum,
        beta=beta,
        ns_steps=ns_steps,
        rms_scale=None,
    )

    assert torch.allclose(update, reference_update, atol=1e-4, rtol=1e-3)
    assert torch.allclose(momentum, reference_momentum, atol=1e-5, rtol=1e-3)
