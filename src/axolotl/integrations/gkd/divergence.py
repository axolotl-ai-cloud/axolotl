# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Axis A (divergence) seam. ``generalized_jsd_loss`` is the GKD core
(https://huggingface.co/papers/2306.13649): ``beta=0`` forward-KL, ``beta=1``
reverse-KL, between = Jensen-Shannon. New divergences register into
``DIVERGENCE_REGISTRY`` and are selected via ``gkd_divergence``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

# (student_logits, teacher_logits, labels, temperature, num_items_in_batch) -> loss
DivergenceFn = Callable[..., torch.Tensor]


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    num_items_in_batch: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized JSD loss (GKD Eq. 1); ``kl_div`` arg order matches TRL (swapped vs the paper)."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    if beta == 0:
        jsd = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
    elif beta == 1:
        jsd = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )
    else:
        beta_t = torch.tensor(
            beta, dtype=student_log_probs.dtype, device=student_log_probs.device
        )
        mixture = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        jsd = beta * F.kl_div(
            mixture, teacher_log_probs, reduction="none", log_target=True
        ) + (1 - beta) * F.kl_div(
            mixture, student_log_probs, reduction="none", log_target=True
        )

    mask = None if labels is None else labels != -100
    if mask is not None:
        jsd = jsd[mask]

    if num_items_in_batch is not None:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(jsd.device)
        return jsd.sum() / num_items_in_batch
    if reduction == "batchmean":
        # clamp_min(1) avoids 0/0 -> nan when a sample has no unmasked positions.
        denom = mask.sum().clamp_min(1) if mask is not None else max(jsd.size(0), 1)
        return jsd.sum() / denom
    if reduction == "sum":
        return jsd.sum()
    if reduction == "mean":
        return jsd.mean()
    return jsd


# name -> factory(beta) -> divergence callable. Extend by assigning new entries.
DIVERGENCE_REGISTRY: dict[str, Callable[[float], DivergenceFn]] = {
    "jsd": lambda beta: partial(generalized_jsd_loss, beta=beta),
    "generalized_jsd": lambda beta: partial(generalized_jsd_loss, beta=beta),
    "forward_kl": lambda beta: partial(generalized_jsd_loss, beta=0.0),
    "fkl": lambda beta: partial(generalized_jsd_loss, beta=0.0),
    "reverse_kl": lambda beta: partial(generalized_jsd_loss, beta=1.0),
    "rkl": lambda beta: partial(generalized_jsd_loss, beta=1.0),
}


def resolve_divergence(name: str | None, beta: float) -> DivergenceFn:
    """``name=None`` defaults to generalized JSD via ``beta`` (spans fwd/rev-KL and between)."""
    factory = DIVERGENCE_REGISTRY.get(name or "jsd")
    if factory is None:
        raise ValueError(
            f"Unknown gkd_divergence '{name}'. Available: {sorted(DIVERGENCE_REGISTRY)}"
        )
    return factory(beta)
