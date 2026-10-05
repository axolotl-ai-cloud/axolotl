"""Source-audited diffusion objective weighting and reductions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from axolotl.model_support.diffusion import ObjectiveReduction, TimeWeighting


def time_weights(times: torch.Tensor, weighting: TimeWeighting) -> torch.Tensor:
    """Return the audited Dream scalar time factor for each logical example."""

    safe = times.clamp_min(1e-12)
    if weighting is TimeWeighting.INV_T:
        return safe.reciprocal()
    if weighting is TimeWeighting.LINEAR:
        return 1.0 - times
    if weighting is TimeWeighting.INV_ONE_MINUS_T:
        return (1.0 - times).clamp_min(1e-12).reciprocal()
    if weighting in {TimeWeighting.NONE, TimeWeighting.CART, TimeWeighting.LOO}:
        return torch.ones_like(times)
    raise ValueError(f"unsupported time weighting: {weighting}")


def cart_weights(unmasked: torch.Tensor, cart_p: float) -> torch.Tensor:
    """Dream CART context weights, evaluated independently per logical sequence."""

    if not 0.0 <= cart_p <= 1.0:
        raise ValueError("cart_p must be in [0, 1]")
    length = unmasked.shape[-1]
    positions = torch.arange(length, device=unmasked.device)
    distance = (positions[:, None] - positions[None, :]).abs()
    if 0.0 < cart_p < 1.0:
        pair = (math.log(cart_p) + (distance - 1) * math.log(1.0 - cart_p)).exp() * 0.5
    else:
        pair = 0.5 * cart_p * (1.0 - cart_p) ** (distance - 1).clamp_min(0)
    pair.fill_diagonal_(0)
    return unmasked.to(pair.dtype).matmul(pair)


def focal_weighted_nll(nll: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Dream's optional token-level focal transformation."""

    return alpha * (1.0 - torch.exp(-nll)).pow(gamma) * nll


def reduce_objective(
    token_loss: torch.Tensor,
    support: torch.Tensor,
    reduction: ObjectiveReduction,
    *,
    logical_ids: torch.Tensor | None = None,
    logical_count: int | None = None,
    denominator: torch.Tensor | float | int | None = None,
) -> torch.Tensor:
    """Reduce without letting physical packed rows alter logical objective weights."""

    token_loss = token_loss.reshape(-1)
    support = support.reshape(-1)
    if logical_ids is not None:
        logical_ids = logical_ids.reshape(-1)
    weighted = token_loss * support.to(token_loss.dtype)
    if reduction in {
        ObjectiveReduction.SUPERVISED_TOKEN_MEAN,
        ObjectiveReduction.MASKED_TOKEN_MEAN,
    }:
        denom = support.sum() if denominator is None else denominator
        return weighted.sum() / torch.as_tensor(
            denom, device=weighted.device, dtype=weighted.dtype
        ).clamp_min(torch.finfo(weighted.dtype).eps)
    if reduction is ObjectiveReduction.EXAMPLE_MEAN:
        if logical_ids is None or logical_count is None:
            raise ValueError("example_mean requires logical_ids and logical_count")
        totals = torch.zeros(
            logical_count, device=token_loss.device, dtype=token_loss.dtype
        )
        counts = torch.zeros_like(totals)
        valid = logical_ids >= 0
        totals.scatter_add_(0, logical_ids[valid], weighted[valid])
        counts.scatter_add_(0, logical_ids[valid], support[valid].to(token_loss.dtype))
        per_example = totals / counts.clamp_min(1)
        if denominator is None:
            return per_example.mean()
        return per_example.sum() / torch.as_tensor(
            denominator, device=per_example.device, dtype=per_example.dtype
        ).clamp_min(torch.finfo(per_example.dtype).eps)
    raise ValueError(f"unsupported objective reduction: {reduction}")


def rhine_loo_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    noisy_ids: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Rhine's LOO logit correction, followed by unreduced clean-target NLL."""

    vocab_size = logits.shape[-1]
    correction = torch.log1p(
        vocab_size * (1.0 - times).clamp_min(0) / times.clamp_min(1e-12)
    )
    if correction.shape != noisy_ids.shape:
        correction = correction.reshape(-1, 1).expand_as(noisy_ids)
    adjusted = logits.float().clone()
    adjusted.scatter_add_(-1, noisy_ids[..., None], correction[..., None])
    return F.cross_entropy(
        adjusted.flatten(0, -2), targets.flatten(), reduction="none"
    ).reshape(targets.shape)
