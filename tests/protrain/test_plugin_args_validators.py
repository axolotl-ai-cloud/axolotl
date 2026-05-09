"""Unit tests for ``ProTrainArgs`` model-level mutex validators.

The plugin refuses to coexist with a handful of Axolotl features (see
``ProTrainArgs._reject_incompatible_features`` for the full list). These
tests construct minimal config dicts and assert Pydantic raises a
``ValidationError`` at load time — catching misconfigurations before the
training loop starts rather than deep inside the chunk manager.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from axolotl.integrations.protrain.args import ProTrainArgs


def _minimal_active_cfg(**overrides) -> dict:
    """A ProTrain-active config that is otherwise valid.

    Base plugin + auto_memory + a base_model is the minimal shape the
    other validators (``_require_plugin_registration``,
    ``_require_model_or_adapter``) are happy with. Tests override one
    field at a time to exercise a single mutex path in isolation.
    """
    cfg: dict = {
        "protrain_auto_memory": True,
        "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
        "base_model": "HuggingFaceTB/SmolLM2-135M",
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------


def test_valid_config_passes() -> None:
    """A config without any excluded fields should validate cleanly."""
    cfg = _minimal_active_cfg()
    # No raise.
    ProTrainArgs.model_validate(cfg)


def test_valid_config_with_inactive_protrain_passes() -> None:
    """With ``protrain_auto_memory`` off, every mutex path short-circuits."""
    cfg = {
        "protrain_auto_memory": False,
        # deepspeed present but auto_memory off => must not raise.
        "deepspeed": "/some/config.json",
    }
    ProTrainArgs.model_validate(cfg)


# ---------------------------------------------------------------------
# Mutex rejections
# ---------------------------------------------------------------------


def test_mutex_rejects_deepspeed() -> None:
    cfg = _minimal_active_cfg(deepspeed="/some/ds_config.json")
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "DeepSpeed" in str(exc.value)


def test_mutex_rejects_fsdp() -> None:
    cfg = _minimal_active_cfg(fsdp=["FULL_SHARD"])
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "FSDP" in str(exc.value)


def test_mutex_rejects_fsdp_config() -> None:
    cfg = _minimal_active_cfg(fsdp_config={"sharding_strategy": "FULL_SHARD"})
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "FSDP" in str(exc.value)


def test_mutex_rejects_gradient_checkpointing() -> None:
    cfg = _minimal_active_cfg(gradient_checkpointing=True)
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "gradient_checkpointing" in msg
    # Must be actionable: tell the user how to resolve it.
    assert "false" in msg or "False" in msg


def test_mutex_allows_gradient_checkpointing_false() -> None:
    """``gradient_checkpointing: false`` is the supported path."""
    cfg = _minimal_active_cfg(gradient_checkpointing=False)
    ProTrainArgs.model_validate(cfg)


def test_mutex_rejects_tensor_parallel() -> None:
    cfg = _minimal_active_cfg(tensor_parallel_size=2)
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "tensor_parallel_size" in str(exc.value)


def test_mutex_allows_tensor_parallel_one() -> None:
    """tp=1 is the single-rank default and must not raise."""
    cfg = _minimal_active_cfg(tensor_parallel_size=1)
    ProTrainArgs.model_validate(cfg)


def test_mutex_rejects_context_parallel() -> None:
    cfg = _minimal_active_cfg(context_parallel_size=4)
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "context_parallel_size" in str(exc.value)


def test_mutex_rejects_sequence_parallel() -> None:
    cfg = _minimal_active_cfg(sequence_parallel_degree=4)
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "sequence_parallel_degree" in str(exc.value)


def test_mutex_allows_load_in_8bit() -> None:
    """M0 spike validated bnb 8-bit composes with ProTrain Mode A; validator must allow it."""
    cfg = _minimal_active_cfg(load_in_8bit=True)
    ProTrainArgs.model_validate(cfg)


def test_mutex_allows_load_in_4bit() -> None:
    """M0 spike validated bnb 4-bit (QLoRA) composes with ProTrain Mode A; validator must allow it."""
    cfg = _minimal_active_cfg(load_in_4bit=True)
    ProTrainArgs.model_validate(cfg)


def test_mutex_allows_load_in_xbit_false() -> None:
    """Both bnb flags explicitly False is still the supported path."""
    cfg = _minimal_active_cfg(load_in_8bit=False, load_in_4bit=False)
    ProTrainArgs.model_validate(cfg)


# ---------------------------------------------------------------------
# Other validators — exercised by proxy, but worth pinning here.
# ---------------------------------------------------------------------


def test_requires_plugin_registration() -> None:
    """``protrain_auto_memory: true`` without the plugin registered fails."""
    cfg = {
        "protrain_auto_memory": True,
        "plugins": [],  # no protrain entry
        "base_model": "foo",
    }
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "plugins" in str(exc.value)


def test_force_all_persistent_default_is_false() -> None:
    """Default for ``protrain_force_all_persistent`` must be False (FIX 2).

    The paper's 4-knob searcher IS the contribution; shipping with it
    disabled by default would hide the feature.
    """
    args = ProTrainArgs()
    assert args.protrain_force_all_persistent is False
