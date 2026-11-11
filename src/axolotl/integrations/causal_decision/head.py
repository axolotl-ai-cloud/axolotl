"""Letter-logit decision head, soft-target loss and per-type temperature fitting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

HEAD_FILENAME = "decision_head.pt"


class DecisionHead(nn.Module):
    """``lm_head`` rows for the answer letters plus a zero-initialized low-rank correction."""

    def __init__(self, rows: torch.Tensor, rank: int, softcap: float | None = None):
        super().__init__()
        hidden_size = rows.shape[1]
        self.weight = nn.Parameter(rows.detach().float().clone())
        self.down = nn.Parameter(
            torch.randn(hidden_size, rank, device=rows.device) / math.sqrt(hidden_size)
        )
        self.up = nn.Parameter(torch.zeros(rank, rows.shape[0], device=rows.device))
        self.softcap = softcap

    @classmethod
    def from_lm_head(
        cls, model: nn.Module, token_ids: Sequence[int], rank: int
    ) -> DecisionHead:
        rows = model.get_output_embeddings().weight[list(token_ids)]
        softcap = getattr(
            model.config.get_text_config(), "final_logit_softcapping", None
        )
        return cls(rows, rank, softcap)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        logits = hidden @ self.weight.T + hidden @ self.down @ self.up
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits


def decision_logits(
    model: nn.Module, head: DecisionHead, lm_head: nn.Module, **inputs
) -> tuple[torch.Tensor, object]:
    """Run ``model`` and apply ``head`` to the final hidden state at the last position.

    Batches are left-padded, so the last position is every row's answer slot.
    """
    hidden = []
    handle = lm_head.register_forward_pre_hook(
        lambda _, args: hidden.append(args[0][:, -1])
    )
    try:
        outputs = model(**inputs, logits_to_keep=1)
    finally:
        handle.remove()
    return head(hidden[-1]), outputs


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean CE against soft ``targets`` over the options ``mask`` allows."""
    logprobs = logits.float().masked_fill(~mask, float("-inf")).log_softmax(-1)
    return -(targets * logprobs.masked_fill(~mask, 0.0)).sum(-1).mean()


def fit_temperature(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, steps: int = 100
) -> float:
    """Temperature minimizing soft-target NLL of ``logits / T``."""
    logits = logits.detach().float()
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=steps, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = soft_cross_entropy(logits / log_temperature.exp(), targets, mask)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.exp().item()


def fit_temperatures(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    groups: Sequence[str],
) -> dict[str, float]:
    """One temperature per question type in ``groups`` (aligned with the rows)."""
    temperatures = {}
    for group in sorted(set(groups)):
        rows = torch.tensor([i for i, name in enumerate(groups) if name == group])
        temperatures[group] = fit_temperature(logits[rows], targets[rows], mask[rows])
    return temperatures
