"""SCOPE-RL: entropy control via temperature-adaptive positive samples (arXiv:2510.08141)."""

import torch


def scope_temperature(
    entropy: float, target_entropy: float, t_min: float, t_max: float
) -> float:
    """Auxiliary sampling temperature ``T = clip(1 + H0 - H(pi_old), t_min, t_max)``."""
    return min(max(1.0 + target_entropy - entropy, t_min), t_max)


def scope_aux_indices(
    num_rows: int, num_generations: int, alpha: float, seed: int
) -> list[int]:
    """Row indices of the rollout groups to resample for the auxiliary branch.

    Whole groups are drawn so downstream ``view(-1, num_generations)`` reshapes stay valid.
    """
    num_groups = num_rows // num_generations
    if num_groups == 0 or alpha <= 0:
        return []
    num_aux = min(num_groups, max(1, round(alpha * num_groups)))
    generator = torch.Generator().manual_seed(seed)
    groups = torch.randperm(num_groups, generator=generator)[:num_aux].sort().values
    return [
        g * num_generations + i for g in groups.tolist() for i in range(num_generations)
    ]


def scope_weights(scope_mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-row loss weights folding the ``alpha`` auxiliary term into a single mean.

    Aggregating ``weights * per_sequence_loss`` over all rows then reproduces
    ``mean(main) + alpha * mean(aux)`` from Eq. 11.
    """
    num_aux = scope_mask.sum()
    num_main = scope_mask.numel() - num_aux
    return torch.where(
        scope_mask.bool(),
        alpha * scope_mask.numel() / num_aux.clamp(min=1),
        scope_mask.numel() / num_main.clamp(min=1),
    )
