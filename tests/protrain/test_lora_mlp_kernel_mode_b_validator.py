"""Unit tests for the ``lora_mlp_kernel`` + Mode B/C validator and the
inert-plugin (``protrain_auto_memory: false``) warning.

Both checks are config-completeness guards that mirror
``_reject_ddp_with_zero3_shard`` (commit ``342e1bd90``): detect a known-bad
composition at config time and either fail or warn loudly with an actionable
message. The kernel/Mode-B/C composition crashes at the first backward pass
with a ``LoRA_MLPBackward`` gradient-shape mismatch (v61, proposal §6.qq);
the inert-plugin case silently runs as vanilla axolotl (audit of v15-v52,
proposal §6.pp).
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from axolotl.integrations.protrain import plugin as protrain_plugin
from axolotl.integrations.protrain.args import ProTrainArgs


def _minimal_active_cfg(**overrides) -> dict:
    """A ProTrain-active config that is otherwise valid."""
    cfg: dict = {
        "protrain_auto_memory": True,
        "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
        "base_model": "HuggingFaceTB/SmolLM2-135M",
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------
# lora_mlp_kernel + Mode B/C — hard rejections
# ---------------------------------------------------------------------


def test_reject_lora_kernel_with_zero3_shard() -> None:
    """Deterministic Mode-C force + fused kernel = guaranteed backward crash."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_zero3_shard=True,
        lora_mlp_kernel=True,
    )
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "lora_mlp_kernel" in msg
    assert "Mode B" in msg or "Mode C" in msg
    assert "LoRA_MLPBackward" in msg
    # Must give the user both fix options.
    assert "lora_mlp_kernel: false" in msg
    assert "protrain_auto_mode: true" in msg


def test_reject_lora_kernel_with_force_replicated_cpu_offload() -> None:
    """Deterministic Mode-B force + fused kernel = guaranteed backward crash."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_replicated_cpu_offload=True,
        lora_mlp_kernel=True,
    )
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "lora_mlp_kernel" in msg
    assert "LoRA_MLPBackward" in msg


# ---------------------------------------------------------------------
# lora_mlp_kernel + auto_mode — soft warning
# ---------------------------------------------------------------------


def test_warn_lora_kernel_with_auto_mode() -> None:
    """Auto-mode might pick Mode B; warn so user can pre-set lora_mlp_kernel=false."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=True,
        lora_mlp_kernel=True,
    )
    with pytest.warns(UserWarning, match="lora_mlp_kernel.*protrain_auto_mode"):
        ProTrainArgs.model_validate(cfg)


def test_accept_lora_kernel_with_mode_a() -> None:
    """Mode A (all-persistent) keeps weights GPU-resident — kernel is safe.

    Explicit ``protrain_force_all_persistent: true`` suppresses the auto-mode
    warning because the user has pinned the safe mode.
    """
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_all_persistent=True,
        lora_mlp_kernel=True,
    )
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ProTrainArgs.model_validate(cfg)


def test_accept_lora_kernel_with_auto_mode_and_force_all_persistent() -> None:
    """Auto-mode + Mode-A force flag — auto-mode wins for selection but force flag still suppresses the warning."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=True,
        protrain_force_all_persistent=True,
        lora_mlp_kernel=True,
    )
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ProTrainArgs.model_validate(cfg)


def test_accept_lora_kernel_without_protrain() -> None:
    """Without ProTrain active the kernel is fine and the validator must short-circuit."""
    cfg = {
        "protrain_auto_memory": False,
        "lora_mlp_kernel": True,
    }
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ProTrainArgs.model_validate(cfg)


def test_accept_lora_kernel_false_under_mode_c() -> None:
    """The validator targets ``lora_mlp_kernel: true`` only; false must pass."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_zero3_shard=True,
        lora_mlp_kernel=False,
    )
    ProTrainArgs.model_validate(cfg)


# ---------------------------------------------------------------------
# Inert-plugin (auto_memory off) warning
# ---------------------------------------------------------------------


class _FakeCfg:
    """Minimal stand-in for the merged axolotl cfg object expected by plugin hooks."""

    def __init__(self, *, plugins, protrain_auto_memory):
        self.plugins = plugins
        self.protrain_auto_memory = protrain_auto_memory


@pytest.fixture(autouse=True)
def _reset_inert_warn_flag(monkeypatch):
    """Each test starts with a fresh one-shot flag so warnings are observable."""
    monkeypatch.setattr(protrain_plugin, "_INERT_AUTO_MEMORY_WARN_FIRED", False)
    yield


def test_warn_when_auto_memory_false_with_plugin_listed(caplog) -> None:
    cfg = _FakeCfg(
        plugins=["axolotl.integrations.protrain.ProTrainPlugin"],
        protrain_auto_memory=False,
    )
    with caplog.at_level(logging.WARNING, logger="axolotl.integrations.protrain.plugin"):
        protrain_plugin._maybe_warn_inert_plugin(cfg)
    assert any(
        "protrain_auto_memory" in record.message
        and "NOT" in record.message
        and record.levelno == logging.WARNING
        for record in caplog.records
    ), f"expected inert-plugin WARN, got: {[r.message for r in caplog.records]}"


def test_warn_fires_at_most_once_per_session(caplog) -> None:
    cfg = _FakeCfg(
        plugins=["axolotl.integrations.protrain.ProTrainPlugin"],
        protrain_auto_memory=False,
    )
    with caplog.at_level(logging.WARNING, logger="axolotl.integrations.protrain.plugin"):
        protrain_plugin._maybe_warn_inert_plugin(cfg)
        protrain_plugin._maybe_warn_inert_plugin(cfg)
        protrain_plugin._maybe_warn_inert_plugin(cfg)
    inert_warns = [
        r for r in caplog.records
        if "protrain_auto_memory" in r.message and r.levelno == logging.WARNING
    ]
    assert len(inert_warns) == 1, (
        f"expected exactly one inert-plugin WARN per session; got {len(inert_warns)}"
    )


def test_no_warn_when_auto_memory_true(caplog) -> None:
    cfg = _FakeCfg(
        plugins=["axolotl.integrations.protrain.ProTrainPlugin"],
        protrain_auto_memory=True,
    )
    with caplog.at_level(logging.WARNING, logger="axolotl.integrations.protrain.plugin"):
        protrain_plugin._maybe_warn_inert_plugin(cfg)
    inert_warns = [
        r for r in caplog.records if "protrain_auto_memory" in r.message
    ]
    assert inert_warns == [], (
        f"expected no inert-plugin WARN under correct config; got: {[r.message for r in inert_warns]}"
    )


def test_no_warn_when_plugin_not_listed(caplog) -> None:
    cfg = _FakeCfg(plugins=[], protrain_auto_memory=False)
    with caplog.at_level(logging.WARNING, logger="axolotl.integrations.protrain.plugin"):
        protrain_plugin._maybe_warn_inert_plugin(cfg)
    inert_warns = [
        r for r in caplog.records if "protrain_auto_memory" in r.message
    ]
    assert inert_warns == [], (
        f"plugin not listed; expected no WARN, got: {[r.message for r in inert_warns]}"
    )
