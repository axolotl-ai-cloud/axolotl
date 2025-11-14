"""Muon math utilities (Newton–Schulz orthogonalization, updates)."""

from __future__ import annotations

import math

import torch

NEWTON_A = 3.4445
NEWTON_B = -4.7750
NEWTON_C = 2.0315


def newton_schulz_orthogonalize(
    matrix: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Approximate the orthogonal factor of `matrix` using Newton–Schulz iterations.
    """

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {matrix.shape}")

    X = matrix.detach().float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    denom = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = X / denom

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = NEWTON_B * A + NEWTON_C * (A @ A)
        X = NEWTON_A * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)
    return X.to(matrix.dtype)


def muon_orthogonal_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float,
    ns_steps: int,
    rms_scale: float | None = None,
) -> torch.Tensor:
    """
    Compute the Muon orthogonalization update using the given gradient and momentum buffer.
    """

    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta)

    original_shape = update.shape
    update_2d = update.view(update.shape[0], -1)
    update_2d = newton_schulz_orthogonalize(update_2d, steps=ns_steps)

    scale = rms_scale
    if scale is None:
        scale = math.sqrt(max(update_2d.size(0), update_2d.size(1)) * 0.4)
    update_2d = update_2d * scale

    return update_2d.view(original_shape)
