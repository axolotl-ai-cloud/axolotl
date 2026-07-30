"""Quantization onto the IQ1_S ternary-pattern codebook.

Groups of 8 consecutive weights along the input dimension jointly snap to the
nearest of the grid's 2048 ternary patterns (measured: +5.1% L2 over free
ternary on FP weights). ~70% of groups' unconstrained ternary rounding is
already a grid pattern, so the exact 2048-way search only runs on the misses.
"""

from __future__ import annotations

import torch

from .iq1s_grid import GRID_DIM, grid_tensor, pattern_index_table


def project_codes(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Nearest grid pattern per group of 8, as ternary codes shaped like `weight`.

    The scalar pre-round prunes the search: groups whose free ternary rounding is
    a grid pattern take it directly (the free pattern is then also the grid
    optimum, since the grid is a subset of the free patterns).
    """
    if weight.shape[-1] % GRID_DIM:
        raise ValueError(
            f"iq1s needs the last dim divisible by {GRID_DIM}, got {weight.shape}"
        )
    with torch.no_grad():
        normalized = (weight.float() / scale.float().clamp_min(1e-12)).reshape(
            -1, GRID_DIM
        )
        free = normalized.round().clamp_(-1, 1)

        powers = torch.tensor(
            [3**i for i in range(GRID_DIM)], dtype=torch.long, device=weight.device
        )
        keys = ((free.long() + 1) * powers).sum(dim=1)
        indices = pattern_index_table(weight.device)[keys]
        misses = indices < 0
        if bool(misses.any()):
            grid = grid_tensor(weight.device).float()
            scores = normalized[misses] @ grid.T * 2 - (grid * grid).sum(dim=1)
            indices[misses] = scores.argmax(dim=1)
        codes = grid_tensor(weight.device)[indices].to(weight.dtype)
        return codes.reshape(weight.shape)


def fake_quant_weight_iq1s(
    weight: torch.Tensor, lambda_: float, scale: torch.Tensor
) -> torch.Tensor:
    """`(1 - λ) · w + λ · pattern · s`, no gradient."""
    codes = project_codes(weight, scale)
    quantized = (codes.float() * scale.float()).to(weight.dtype)
    if lambda_ >= 1.0:
        return quantized
    return ((1.0 - lambda_) * weight.float() + lambda_ * quantized.float()).to(
        weight.dtype
    )


class FakeQuantWeightIQ1SSTE(torch.autograd.Function):
    """Straight-through estimator over the grid projection.

    The weight gradient passes straight through (the projection is
    piecewise-constant); a learnable scale receives `Σ grad · code`, the exact
    gradient of `code · s` holding the assignment fixed — the same convention
    the single-plane learnable ternary STE uses.
    """

    @staticmethod
    def forward(ctx, weight, lambda_, scale):  # type: ignore[override]
        codes = project_codes(weight, scale)
        ctx.save_for_backward(codes)
        ctx.lambda_ = lambda_
        ctx.scale_needs_grad = scale.requires_grad
        quantized = (codes.float() * scale.float()).to(weight.dtype)
        if lambda_ >= 1.0:
            return quantized
        return ((1.0 - lambda_) * weight.float() + lambda_ * quantized.float()).to(
            weight.dtype
        )

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        (codes,) = ctx.saved_tensors
        grad_scale = None
        if ctx.scale_needs_grad:
            grad_scale = ((grad.float() * codes.float()).sum() * ctx.lambda_).reshape(1)
        return grad, None, grad_scale


def fake_quant_weight_iq1s_ste(
    weight: torch.Tensor, lambda_: float, scale: torch.Tensor
) -> torch.Tensor:
    """STE entry point mirroring the other codebooks' wrappers."""
    return FakeQuantWeightIQ1SSTE.apply(weight, lambda_, scale)
