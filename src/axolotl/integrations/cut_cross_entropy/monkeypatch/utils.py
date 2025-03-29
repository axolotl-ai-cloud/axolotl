# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""Monkeypatch for apply_lce to add softcap."""

import torch
from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.transformers.utils import PatchOptions


def apply_lce(
    e: torch.Tensor,
    c: torch.Tensor,
    labels: torch.Tensor,
    opts: PatchOptions,
    bias: torch.Tensor | None = None,
    softcap: float | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    """Monkey patch for apply_lce to support softcap kwarg."""
    num_items_in_batch = loss_kwargs.get("num_items_in_batch", None)
    cce_kwargs = opts.to_kwargs()
    if num_items_in_batch is not None and cce_kwargs["reduction"] == "mean":
        cce_kwargs["reduction"] = "sum"
    else:
        num_items_in_batch = None

    loss = linear_cross_entropy(
        e,
        c,
        labels.to(e.device),
        bias=bias,
        shift=True,
        softcap=softcap,
        **cce_kwargs,
    )

    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch

    return loss
