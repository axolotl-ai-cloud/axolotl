"""Mode A ``force_all_persistent`` + full-FT + multi-rank DDP setup guard.

Mode A's chunk-wrapper hooks register per-param autograd hooks. With every
parameter trainable (full finetune) and DDP active (world_size > 1), those
hooks collide with DDP's reducer and surface much later as "parameters which
did not receive grad" deep in the backward pass. The plugin's
``post_trainer_create`` raises a clear ``RuntimeError`` at setup time so
users don't have to chase the cryptic DDP failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from axolotl.integrations.protrain.plugin import (
    _guard_force_all_persistent_full_ft_ddp,
)


class _FullFTModel:
    """Minimal model stub with all trainable params (mimics a full-FT model)."""

    def __init__(self, n_params: int = 3, all_trainable: bool = True) -> None:
        import torch

        self._params = [
            torch.nn.Parameter(torch.zeros(2), requires_grad=all_trainable)
            for _ in range(n_params)
        ]

    def parameters(self):
        return iter(self._params)


class _PartialFTModel:
    """Model with mixed trainable/frozen params (mimics LoRA: most frozen)."""

    def __init__(self) -> None:
        import torch

        self._params = [
            torch.nn.Parameter(torch.zeros(2), requires_grad=True),
            torch.nn.Parameter(torch.zeros(2), requires_grad=False),
            torch.nn.Parameter(torch.zeros(2), requires_grad=False),
        ]

    def parameters(self):
        return iter(self._params)


@pytest.fixture
def _world_size_env(monkeypatch):
    """Yield a setter for WORLD_SIZE that auto-clears on teardown."""

    def _set(value: int | None) -> None:
        if value is None:
            monkeypatch.delenv("WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("WORLD_SIZE", str(value))

    return _set


def test_guard_raises_on_force_all_persistent_full_ft_multi_rank(_world_size_env):
    """All knobs aligned with the bad combination → RuntimeError with actionable text."""
    _world_size_env(4)
    cfg = SimpleNamespace(protrain_force_all_persistent=True)
    trainer = SimpleNamespace(model=_FullFTModel(all_trainable=True))

    with pytest.raises(RuntimeError) as exc:
        _guard_force_all_persistent_full_ft_ddp(cfg, trainer)

    msg = str(exc.value)
    assert "force_all_persistent" in msg
    assert "full finetune" in msg
    assert "multi-rank DDP" in msg
    # Must point users at the supported alternative for full-FT.
    assert "protrain_zero3_shard" in msg


def test_guard_noop_when_force_all_persistent_off(_world_size_env):
    """Default (force_all_persistent=False) short-circuits before any other check."""
    _world_size_env(4)
    cfg = SimpleNamespace(protrain_force_all_persistent=False)
    trainer = SimpleNamespace(model=_FullFTModel(all_trainable=True))
    _guard_force_all_persistent_full_ft_ddp(cfg, trainer)  # no raise


def test_guard_noop_on_single_gpu(_world_size_env):
    """World size <= 1 is fine — there's no DDP reducer to collide with."""
    _world_size_env(1)
    cfg = SimpleNamespace(protrain_force_all_persistent=True)
    trainer = SimpleNamespace(model=_FullFTModel(all_trainable=True))
    _guard_force_all_persistent_full_ft_ddp(cfg, trainer)  # no raise


def test_guard_noop_when_some_params_frozen(_world_size_env):
    """LoRA / QLoRA (most params frozen) is the supported Mode A workload."""
    _world_size_env(4)
    cfg = SimpleNamespace(protrain_force_all_persistent=True)
    trainer = SimpleNamespace(model=_PartialFTModel())
    _guard_force_all_persistent_full_ft_ddp(cfg, trainer)  # no raise


def test_guard_noop_when_model_missing(_world_size_env):
    """No model on trainer (uncommon, but possible mid-construction) → no raise."""
    _world_size_env(4)
    cfg = SimpleNamespace(protrain_force_all_persistent=True)
    trainer = SimpleNamespace(model=None)
    _guard_force_all_persistent_full_ft_ddp(cfg, trainer)  # no raise


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
