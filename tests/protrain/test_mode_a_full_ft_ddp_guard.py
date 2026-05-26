"""Mode A (all-persistent) + full-FT + multi-rank DDP setup guard.

Mode A's chunk-wrapper hooks register per-param autograd hooks. With every
parameter trainable (full finetune) and DDP active (world_size > 1), those
hooks collide with DDP's reducer and surface much later as "parameters which
did not receive grad" deep in the backward pass. The plugin's
``post_trainer_create`` raises a clear ``RuntimeError`` at setup time so
users don't have to chase the cryptic DDP failure.

Detection uses the EFFECTIVE runtime state (post-searcher
``n_persist >= N_chunk``) rather than the raw cfg flag, so the guard fires
regardless of whether Mode A was selected via ``protrain_force_all_persistent``
or via auto-mode landing on an all-persistent config. Full-FT is detected
via ``cfg.adapter is None/empty`` (config-level, survives ProTrain's model
wrapping).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from axolotl.integrations.protrain.plugin import (
    _guard_force_all_persistent_full_ft_ddp,  # back-compat alias
    _guard_full_ft_mode_a_ddp,
)


def _make_wrapped(n_persist: int, n_chunk: int) -> SimpleNamespace:
    """Synthesize the minimum surface the guard inspects on `wrapped`."""
    return SimpleNamespace(
        search_result=SimpleNamespace(cfg=SimpleNamespace(n_persist=n_persist)),
        chunk_manager=SimpleNamespace(layout=SimpleNamespace(N_chunk=n_chunk)),
    )


def _make_trainer():
    import torch

    return SimpleNamespace(
        model=torch.nn.Linear(4, 4),
        args=SimpleNamespace(),
    )


@pytest.fixture
def _world_size_env(monkeypatch):
    """Yield a setter for WORLD_SIZE that auto-clears on teardown."""

    def _set(value: int | None) -> None:
        if value is None:
            monkeypatch.delenv("WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("WORLD_SIZE", str(value))

    return _set


def test_back_compat_alias_resolves():
    """Old name still importable; refers to the same function."""
    assert _guard_force_all_persistent_full_ft_ddp is _guard_full_ft_mode_a_ddp


def test_full_ft_mode_a_multi_rank_raises(_world_size_env):
    """Full-FT (no adapter) + Mode A (n_persist == N_chunk) + multi-rank → raise."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter=None)
    trainer = _make_trainer()
    wrapped = _make_wrapped(n_persist=315, n_chunk=315)

    with pytest.raises(RuntimeError) as exc:
        _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)

    msg = str(exc.value)
    assert "Mode A" in msg
    assert "full finetune" in msg
    # Must point users at the supported alternative for full-FT.
    assert "protrain_zero3_shard" in msg


def test_full_ft_mode_c_multi_rank_no_raise(_world_size_env):
    """Full-FT + Mode C (n_persist < N_chunk) → no raise (the supported path)."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter=None)
    trainer = _make_trainer()
    wrapped = _make_wrapped(n_persist=103, n_chunk=315)

    _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)  # no raise


@pytest.mark.parametrize("adapter", ["lora", "qlora", "LoRA"])
def test_lora_mode_a_no_raise(_world_size_env, adapter):
    """LoRA / qLoRA workloads are not affected even at Mode A."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter=adapter)
    trainer = _make_trainer()
    wrapped = _make_wrapped(n_persist=315, n_chunk=315)

    _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)  # no raise


def test_single_gpu_no_raise(_world_size_env):
    """World size 1: Mode A is fine because there's no DDP wrap."""
    _world_size_env(1)
    cfg = SimpleNamespace(adapter=None)
    trainer = _make_trainer()
    wrapped = _make_wrapped(n_persist=315, n_chunk=315)

    _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)  # no raise


def test_empty_adapter_string_treated_as_full_ft(_world_size_env):
    """`adapter: ""` from YAML is treated as full-FT, same as None."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter="")
    trainer = _make_trainer()
    wrapped = _make_wrapped(n_persist=315, n_chunk=315)

    with pytest.raises(RuntimeError, match="Mode A"):
        _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)


def test_missing_wrapped_is_defensive(_world_size_env):
    """No wrapped object: silent no-op (post_model_load may have been skipped)."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter=None)
    trainer = _make_trainer()

    _guard_full_ft_mode_a_ddp(cfg, trainer, None)  # no raise


def test_missing_search_result_is_defensive(_world_size_env):
    """Wrapped without search_result attr: silent no-op."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter=None)
    trainer = _make_trainer()
    wrapped = SimpleNamespace(
        chunk_manager=SimpleNamespace(layout=SimpleNamespace(N_chunk=10))
    )

    _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)  # no raise


def test_missing_chunk_manager_is_defensive(_world_size_env):
    """Wrapped without chunk_manager attr: silent no-op."""
    _world_size_env(4)
    cfg = SimpleNamespace(adapter=None)
    trainer = _make_trainer()
    wrapped = SimpleNamespace(
        search_result=SimpleNamespace(cfg=SimpleNamespace(n_persist=10))
    )

    _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)  # no raise


def test_mode_a_via_auto_mode_still_caught(_world_size_env):
    """Even if cfg.protrain_force_all_persistent is False (auto-mode landed
    on Mode A), the guard still fires because it reads runtime state from
    the picked CostConfig, not the user-input flag. This is the Codex review
    finding #1 fix.
    """
    _world_size_env(4)
    cfg = SimpleNamespace(
        adapter=None,
        protrain_force_all_persistent=False,  # the old guard would have skipped here
        protrain_auto_mode=True,
    )
    trainer = _make_trainer()
    wrapped = _make_wrapped(n_persist=315, n_chunk=315)

    with pytest.raises(RuntimeError, match="Mode A"):
        _guard_full_ft_mode_a_ddp(cfg, trainer, wrapped)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
