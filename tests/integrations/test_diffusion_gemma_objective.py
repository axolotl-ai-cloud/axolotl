"""Unit tests for the DiffusionGemma block-diffusion objective.

These tests exercise the corruption process and loss in isolation, without
instantiating the DiffusionGemma model.
"""

import pytest
import torch

from axolotl.integrations.diffusion_gemma.diffusion import (
    DiffusionObjectiveConfig,
    corrupt_canvas,
    diffusion_loss,
    sample_timesteps,
)


def test_sample_timesteps_in_range():
    t = sample_timesteps(1024, torch.device("cpu"), eps=1e-3)
    assert t.shape == (1024,)
    assert (t >= 1e-3).all() and (t <= 1.0).all()


def test_uniform_corruption_fraction_tracks_t():
    """At t≈1 almost everything is corrupted; at t≈0 almost nothing."""
    cfg = DiffusionObjectiveConfig(vocab_size=100)
    x0 = torch.randint(0, 100, (256, 64))
    g = torch.Generator().manual_seed(0)

    t_hi = torch.full((256,), 0.95)
    _, mask_hi = corrupt_canvas(x0, t_hi, cfg, generator=g)
    t_lo = torch.full((256,), 0.05)
    _, mask_lo = corrupt_canvas(x0, t_lo, cfg, generator=g)

    assert mask_hi.float().mean() > 0.85
    assert mask_lo.float().mean() < 0.15


def test_corruption_replaces_only_marked_positions():
    cfg = DiffusionObjectiveConfig(vocab_size=100)
    x0 = torch.randint(0, 100, (8, 32))
    t = torch.full((8,), 0.5)
    noised, mask = corrupt_canvas(x0, t, cfg, generator=torch.Generator().manual_seed(1))
    # Unmarked positions must be unchanged
    assert torch.equal(noised[~mask], x0[~mask])


def test_eligible_mask_excludes_padding():
    """Padding positions (eligible=0) are never corrupted nor counted."""
    cfg = DiffusionObjectiveConfig(vocab_size=100)
    x0 = torch.randint(0, 100, (4, 16))
    eligible = torch.ones_like(x0)
    eligible[:, 8:] = 0  # second half is padding
    t = torch.full((4,), 0.9)
    noised, mask = corrupt_canvas(x0, t, cfg, eligible_mask=eligible)
    assert mask[:, 8:].sum() == 0
    assert torch.equal(noised[:, 8:], x0[:, 8:])


def test_mask_corruption_uses_mask_token():
    cfg = DiffusionObjectiveConfig(vocab_size=100, corruption="mask", mask_token_id=99)
    x0 = torch.randint(0, 99, (4, 16))
    t = torch.full((4,), 1.0)
    noised, mask = corrupt_canvas(x0, t, cfg)
    assert (noised[mask] == 99).all()


def test_mask_corruption_requires_token_id():
    with pytest.raises(ValueError):
        DiffusionObjectiveConfig(vocab_size=100, corruption="mask")


def test_loss_zero_when_logits_perfect():
    """Confident-correct logits on corrupted positions ⇒ ~zero loss."""
    cfg = DiffusionObjectiveConfig(vocab_size=10)
    x0 = torch.randint(0, 10, (4, 16))
    t = torch.full((4,), 0.5)
    _, mask = corrupt_canvas(x0, t, cfg, generator=torch.Generator().manual_seed(2))
    logits = torch.full((4, 16, 10), -10.0)
    logits.scatter_(2, x0.unsqueeze(-1), 10.0)  # one-hot toward x0
    loss, metrics = diffusion_loss(logits, x0, mask, t, cfg)
    assert loss.item() < 1e-3
    assert metrics["diffusion/token_ce"] < 1e-3


def test_loss_elbo_weight_upweights_small_t():
    """Identical per-token CE but smaller t ⇒ larger ELBO-weighted loss."""
    cfg = DiffusionObjectiveConfig(vocab_size=10, loss_weighting="elbo")
    x0 = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.bool)
    logits = torch.zeros(2, 8, 10)  # uniform ⇒ CE = ln(10) everywhere
    loss_small, _ = diffusion_loss(logits, x0, mask, torch.full((2,), 0.1), cfg)
    loss_large, _ = diffusion_loss(logits, x0, mask, torch.full((2,), 0.9), cfg)
    assert loss_small > loss_large


def test_loss_gradient_flows():
    cfg = DiffusionObjectiveConfig(vocab_size=10)
    x0 = torch.randint(0, 10, (4, 16))
    t = torch.full((4,), 0.5)
    _, mask = corrupt_canvas(x0, t, cfg, generator=torch.Generator().manual_seed(3))
    logits = torch.randn(4, 16, 10, requires_grad=True)
    loss, _ = diffusion_loss(logits, x0, mask, t, cfg)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_loss_ignores_examples_without_corruption():
    """An example with no corrupted tokens must not produce NaN."""
    cfg = DiffusionObjectiveConfig(vocab_size=10)
    x0 = torch.randint(0, 10, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, :4] = True  # only first example has targets
    logits = torch.randn(2, 8, 10)
    t = torch.full((2,), 0.5)
    loss, _ = diffusion_loss(logits, x0, mask, t, cfg)
    assert torch.isfinite(loss)
