"""DFT (Dynamic Fine-Tuning) loss utilities."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_per_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token CE loss and the corresponding valid mask.

    Returns:
        per_token_loss: 1D tensor of shape [B * (T - shift)] with grad.
        valid_mask: 1D bool tensor with True where label != ignore_index.
    """
    if shift_labels:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        shift_labels = labels

    shift_logits = shift_logits.float()
    shift_labels = shift_labels.to(shift_logits.device)

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )

    valid_mask = shift_labels.reshape(-1) != ignore_index
    return per_token_loss, valid_mask


def apply_dft_weighting(per_token_loss: torch.Tensor) -> torch.Tensor:
    """Apply DFT weighting without allowing gradients through the weight."""
    with torch.no_grad():
        weights = torch.exp(-per_token_loss)
    return per_token_loss * weights


def reduce_token_loss(
    per_token_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """Reduce masked per-token loss to a scalar."""
    if num_items_in_batch is None:
        denom = valid_mask.sum()
    else:
        denom = per_token_loss.new_tensor(int(num_items_in_batch))

    if int(denom.item()) == 0:
        return per_token_loss.sum() * 0.0

    return per_token_loss[valid_mask].sum() / denom


def compute_dft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """Compute scalar DFT loss from logits + labels."""
    per_token_loss, valid_mask = compute_per_token_cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
    )
    per_token_loss = apply_dft_weighting(per_token_loss)
    return reduce_token_loss(
        per_token_loss,
        valid_mask,
        num_items_in_batch=num_items_in_batch,
    )

