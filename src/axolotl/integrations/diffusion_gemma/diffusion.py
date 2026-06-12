"""Core block-diffusion training objective for DiffusionGemma.

This module is intentionally model-agnostic and depends only on torch, so the
corruption process and loss can be unit-tested without instantiating the 26B
model.

The forward (noising) process mirrors DiffusionGemma's inference sampler
(`EntropyBoundSampler`): the canvas is corrupted by *uniformly resampling*
tokens from the vocabulary, not by an absorbing ``[MASK]`` token. At time
``t in (0, 1]`` each canvas position is independently corrupted with
probability ``t``; the denoiser is trained to recover the clean token at the
corrupted positions. An absorbing (mask-token) variant is also supported for
experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionObjectiveConfig:
    """Hyperparameters for the block-diffusion objective."""

    vocab_size: int
    corruption: str = "uniform"  # "uniform" (matches inference) | "mask"
    mask_token_id: int | None = None
    timestep_eps: float = 1e-3  # avoid t=0 (division by zero in elbo weighting)
    loss_weighting: str = "elbo"  # "elbo" (1/t) | "uniform" (1.0)
    self_conditioning_prob: float = 0.5

    def __post_init__(self):
        if self.corruption not in ("uniform", "mask"):
            raise ValueError(
                f"corruption must be 'uniform' or 'mask', got {self.corruption}"
            )
        if self.corruption == "mask" and self.mask_token_id is None:
            raise ValueError("corruption='mask' requires mask_token_id to be set")
        if self.loss_weighting not in ("elbo", "uniform"):
            raise ValueError(
                f"loss_weighting must be 'elbo' or 'uniform', got {self.loss_weighting}"
            )
        if not 0.0 <= self.self_conditioning_prob <= 1.0:
            raise ValueError("self_conditioning_prob must be in [0, 1]")


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    eps: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample continuous diffusion times ``t`` in ``[eps, 1]``, one per example."""
    t = torch.rand(batch_size, device=device, generator=generator)
    return t * (1.0 - eps) + eps


def corrupt_canvas(
    clean_canvas: torch.Tensor,
    timesteps: torch.Tensor,
    cfg: DiffusionObjectiveConfig,
    eligible_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the forward (noising) process to a clean canvas.

    Args:
        clean_canvas: ``(batch, canvas_len)`` long tensor of target tokens ``x0``.
        timesteps: ``(batch,)`` float tensor of corruption probabilities ``t``.
        cfg: objective config.
        eligible_mask: ``(batch, canvas_len)`` bool/long tensor; only positions
            that are 1 may be corrupted and contribute to the loss (e.g. real
            answer tokens, excluding right-padding). Defaults to all-ones.
        generator: optional RNG for reproducibility.

    Returns:
        ``(noised_canvas, corrupted_mask)`` where ``corrupted_mask`` is a bool
        tensor marking the positions the denoiser must predict.
    """
    batch, canvas_len = clean_canvas.shape
    device = clean_canvas.device
    if eligible_mask is None:
        eligible_mask = torch.ones_like(clean_canvas, dtype=torch.bool)
    else:
        eligible_mask = eligible_mask.bool()

    u = torch.rand(batch, canvas_len, device=device, generator=generator)
    corrupted_mask = (u < timesteps[:, None]) & eligible_mask

    if cfg.corruption == "uniform":
        noise = torch.randint(
            low=0,
            high=cfg.vocab_size,
            size=(batch, canvas_len),
            device=device,
            generator=generator,
            dtype=clean_canvas.dtype,
        )
    else:  # "mask"
        noise = torch.full_like(clean_canvas, cfg.mask_token_id)

    noised_canvas = torch.where(corrupted_mask, noise, clean_canvas)
    return noised_canvas, corrupted_mask


def diffusion_loss(
    logits: torch.Tensor,
    clean_canvas: torch.Tensor,
    corrupted_mask: torch.Tensor,
    timesteps: torch.Tensor,
    cfg: DiffusionObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reweighted cross-entropy on corrupted positions.

    Per example ``i`` the loss is

        w(t_i) * (1 / N_i) * sum_{j corrupted} CE(logits_ij, x0_ij)

    where ``N_i`` is the number of corrupted positions in example ``i`` and
    ``w(t) = 1/t`` for the ELBO weighting (or ``1`` for uniform). The batch loss
    is the mean over examples that have at least one corrupted position.
    """
    batch, canvas_len, vocab = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        clean_canvas.reshape(-1),
        reduction="none",
    ).reshape(batch, canvas_len)

    mask = corrupted_mask.to(ce.dtype)
    per_example_count = mask.sum(dim=-1)
    has_target = per_example_count > 0
    safe_count = per_example_count.clamp(min=1.0)
    per_example_ce = (ce * mask).sum(dim=-1) / safe_count

    if cfg.loss_weighting == "elbo":
        weight = 1.0 / timesteps
    else:
        weight = torch.ones_like(timesteps)

    weighted = per_example_ce * weight * has_target.to(ce.dtype)
    denom = has_target.sum().clamp(min=1)
    loss = weighted.sum() / denom

    with torch.no_grad():
        total_corrupted = per_example_count.sum()
        token_ce = (ce * mask).sum() / total_corrupted.clamp(min=1.0)
        metrics = {
            "diffusion/token_ce": token_ce.item(),
            "diffusion/corrupted_frac": (total_corrupted / (batch * canvas_len)).item(),
            "diffusion/mean_t": timesteps.mean().item(),
        }
    return loss, metrics
