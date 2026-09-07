"""Unit tests for axolotl.monkeypatch.relora.reset_optimizer."""

import math

import pytest
import torch
import torch.nn as nn

from axolotl.monkeypatch.relora import (
    magnitude_pruning_,
    random_pruning_,
    reset_optimizer,
)

ADAM_KEYS = ["exp_avg", "exp_avg_sq"]


def _build_optimizer_with_state(seed: int = 0):
    """Build a tiny optimizer over LoRA-shaped + non-LoRA params with populated state."""
    torch.manual_seed(seed)
    lora_a = nn.Parameter(torch.randn(8, 32))
    lora_b = nn.Parameter(torch.randn(32, 8))
    extra = nn.Parameter(torch.randn(64, 32))

    optimizer = torch.optim.AdamW([lora_a, lora_b, extra], lr=1e-3)
    for _ in range(2):
        loss = (
            (lora_a * torch.randn_like(lora_a)).sum()
            + (lora_b * torch.randn_like(lora_b)).sum()
            + (extra * torch.randn_like(extra)).sum()
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return optimizer, lora_a, lora_b, extra


def test_reset_optimizer_only_touches_reset_params():
    """State for params NOT in reset_params must be byte-identical after reset."""
    optimizer, lora_a, lora_b, extra = _build_optimizer_with_state()

    extra_avg_before = optimizer.state[extra]["exp_avg"].clone()
    extra_avg_sq_before = optimizer.state[extra]["exp_avg_sq"].clone()

    reset_optimizer(
        optimizer,
        reset_params=[lora_a, lora_b],
        optimizer_state_keys=ADAM_KEYS,
        prune_method="magnitude",
        prune_ratio=0.9,
    )

    assert torch.equal(optimizer.state[extra]["exp_avg"], extra_avg_before)
    assert torch.equal(optimizer.state[extra]["exp_avg_sq"], extra_avg_sq_before)


def test_reset_optimizer_actually_prunes_lora_state():
    optimizer, lora_a, lora_b, _extra = _build_optimizer_with_state()

    reset_optimizer(
        optimizer,
        reset_params=[lora_a, lora_b],
        optimizer_state_keys=ADAM_KEYS,
        prune_method="magnitude",
        prune_ratio=0.9,
    )

    for param in (lora_a, lora_b):
        for key in ADAM_KEYS:
            zero_frac = (optimizer.state[param][key] == 0).float().mean().item()
            assert zero_frac >= 0.85


@pytest.mark.parametrize(
    "method,ratio,expected_zero_frac",
    [
        ("magnitude", 0.9, 0.9),
        ("magnitude", 0.99, 0.99),
        ("random", 0.9, 0.9),
        ("random", 0.5, 0.5),
        # reset uses random pruning; relora_prune_ratio must be honored, not ignored.
        ("reset", 0.9, 0.9),
        ("reset", 0.5, 0.5),
    ],
)
def test_prune_methods(method, ratio, expected_zero_frac):
    """Each method zeros approximately the expected fraction."""
    optimizer, lora_a, lora_b, _extra = _build_optimizer_with_state(seed=42)

    reset_optimizer(
        optimizer,
        reset_params=[lora_a, lora_b],
        optimizer_state_keys=ADAM_KEYS,
        prune_method=method,
        prune_ratio=ratio,
    )

    total = 0
    zeros = 0
    for param in (lora_a, lora_b):
        for key in ADAM_KEYS:
            tensor = optimizer.state[param][key]
            total += tensor.numel()
            zeros += (tensor == 0).sum().item()

    actual = zeros / total
    tolerance = 0.02 if method == "magnitude" else 0.05
    assert math.isclose(actual, expected_zero_frac, abs_tol=tolerance)


def test_reset_optimizer_skips_keys_not_in_state_keys():
    """Keys present in optimizer state but not in optimizer_state_keys are untouched."""
    optimizer, lora_a, lora_b, _extra = _build_optimizer_with_state()

    exp_avg_sq_before = optimizer.state[lora_a]["exp_avg_sq"].clone()

    reset_optimizer(
        optimizer,
        reset_params=[lora_a, lora_b],
        optimizer_state_keys=["exp_avg"],
        prune_method="magnitude",
        prune_ratio=0.9,
    )

    assert torch.equal(optimizer.state[lora_a]["exp_avg_sq"], exp_avg_sq_before)


def test_reset_optimizer_handles_param_with_empty_state():
    """Params with no optimizer state are skipped silently."""
    optimizer, lora_a, lora_b, _extra = _build_optimizer_with_state()
    orphan = nn.Parameter(torch.randn(4, 4))

    reset_optimizer(
        optimizer,
        reset_params=[lora_a, lora_b, orphan],
        optimizer_state_keys=ADAM_KEYS,
        prune_method="magnitude",
        prune_ratio=0.9,
    )

    assert orphan not in optimizer.state or not optimizer.state[orphan]


def test_unknown_prune_method_raises():
    optimizer, lora_a, lora_b, _extra = _build_optimizer_with_state()

    with pytest.raises(ValueError, match="Unknown prune_method"):
        reset_optimizer(
            optimizer,
            reset_params=[lora_a, lora_b],
            optimizer_state_keys=ADAM_KEYS,
            prune_method="bogus",  # type: ignore[arg-type]
            prune_ratio=0.9,
        )


def test_pruning_helpers_are_inplace():
    """magnitude_pruning_ and random_pruning_ must mutate via tensor.mul_."""
    tensor = torch.randn(64)
    ptr_before = tensor.data_ptr()
    magnitude_pruning_(tensor, 0.5)
    assert tensor.data_ptr() == ptr_before

    tensor = torch.randn(64)
    ptr_before = tensor.data_ptr()
    random_pruning_(tensor, 0.5)
    assert tensor.data_ptr() == ptr_before


def test_pruning_helpers_support_uint8_tensors():
    """Both pruning helpers must work on uint8 optimizer state tensors."""
    tensor = torch.arange(1, 129, dtype=torch.uint8)
    magnitude_pruning_(tensor, 0.9)

    assert tensor.dtype == torch.uint8
    magnitude_zero_frac = (tensor == 0).float().mean().item()
    assert 0.85 <= magnitude_zero_frac <= 0.95

    tensor = torch.arange(1, 129, dtype=torch.uint8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        random_pruning_(tensor, 0.9)

    assert tensor.dtype == torch.uint8
    random_zero_frac = (tensor == 0).float().mean().item()
    assert 0.85 <= random_zero_frac <= 0.95
