"""Source-audited diffusion objective weighting and reductions."""

from __future__ import annotations

import torch

from axolotl.model_support.diffusion import ObjectiveReduction, TimeWeighting


def time_weights(times: torch.Tensor, weighting: TimeWeighting) -> torch.Tensor:
    """Return the scalar time factor for each logical example."""

    safe = times.clamp_min(1e-12)
    if weighting is TimeWeighting.INV_T:
        return safe.reciprocal()
    if weighting is TimeWeighting.LINEAR:
        return 1.0 - times
    if weighting is TimeWeighting.NONE:
        return torch.ones_like(times)
    raise ValueError(f"unsupported time weighting: {weighting}")


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
