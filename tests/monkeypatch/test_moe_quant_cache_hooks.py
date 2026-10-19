"""Tests for the MoE 8-bit parametrization cache-gate hooks."""

import torch
import torch.nn.utils.parametrize as P
from torch import nn
from torch.utils.checkpoint import checkpoint

from axolotl.monkeypatch.moe_quant import (
    _disable_parametrization_cache,
    _register_parametrization_cache_hooks,
)


class _ResidualBlock(nn.Module):
    """Minimal stand-in for a MoE decoder layer: experts, then a residual add.

    Under ``use_reentrant=False``, early stop fires once the last tensor saved for
    backward has been rematerialized. A residual add saves nothing, so the last save
    lands inside the expert module's forward and its forward hook is skipped. A
    trailing op that does save (another ``Linear``) hides this.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.Linear(8, 8)
        self.experts = nn.Linear(8, 8)

    def forward(self, x):
        residual = self.norm(x)
        return residual + self.experts(residual)


def _reset_cache():
    P._cache_enabled = 0
    P._cache = {}


class TestParametrizationCacheHooks:
    """The cache-gate hook pair must stay balanced under activation checkpointing.

    The pre-hook increments the global ``parametrize._cache_enabled`` counter and the
    post-hook decrements it, clearing the cache at zero. When the post-hook is skipped
    the counter leaks upward once per checkpointed layer per step, the cache is never
    cleared again, and every dequantized expert stays resident for the rest of training.
    """

    def test_counter_balanced_under_checkpoint_early_stop(self):
        _reset_cache()
        try:
            model = _ResidualBlock()
            _register_parametrization_cache_hooks(model.experts)

            for _ in range(3):
                inputs = torch.randn(2, 8, requires_grad=True)
                checkpoint(model, inputs, use_reentrant=False).sum().backward()

            assert P._cache_enabled == 0, (
                f"parametrize cache counter leaked to {P._cache_enabled}; dequantized "
                "expert weights would be retained for the rest of training"
            )
            assert P._cache == {}
        finally:
            _reset_cache()

    def test_counter_never_goes_negative(self):
        _reset_cache()
        try:
            P._cache[("k",)] = torch.zeros(1)
            _disable_parametrization_cache(nn.Identity(), (), None)

            assert P._cache_enabled == 0
            assert P._cache == {}
        finally:
            _reset_cache()
