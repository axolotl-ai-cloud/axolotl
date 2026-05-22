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


# ---------------------------------------------------------------------
# Optimizer allow-list (M6B) — ProTrain's chunk-manager adapters only
# drive AdamW-shaped state. Unsupported optimizers must be rejected at
# config-load time rather than corrupting state inside the step path.
# ---------------------------------------------------------------------


def test_optimizer_validator_accepts_adamw_torch() -> None:
    cfg = _minimal_active_cfg(optimizer="adamw_torch")
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_accepts_adamw_torch_fused() -> None:
    cfg = _minimal_active_cfg(optimizer="adamw_torch_fused")
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_accepts_adamw_8bit() -> None:
    cfg = _minimal_active_cfg(optimizer="adamw_8bit")
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_accepts_adamw_bnb_8bit() -> None:
    cfg = _minimal_active_cfg(optimizer="adamw_bnb_8bit")
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_accepts_paged_adamw_8bit() -> None:
    cfg = _minimal_active_cfg(optimizer="paged_adamw_8bit")
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_accepts_missing_optimizer() -> None:
    """No ``optimizer`` key — Axolotl picks a supported default elsewhere."""
    cfg = _minimal_active_cfg()
    assert "optimizer" not in cfg
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_accepts_none_optimizer() -> None:
    """Explicit ``optimizer: null`` must not raise (default-fill happens later)."""
    cfg = _minimal_active_cfg(optimizer=None)
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_rejects_lion() -> None:
    cfg = _minimal_active_cfg(optimizer="lion_pytorch")
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "lion_pytorch" in msg
    assert "ProTrain" in msg


def test_optimizer_validator_rejects_adafactor() -> None:
    cfg = _minimal_active_cfg(optimizer="adafactor")
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "adafactor" in str(exc.value)


def test_optimizer_validator_rejects_sgd() -> None:
    cfg = _minimal_active_cfg(optimizer="sgd")
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    assert "sgd" in str(exc.value)


def test_optimizer_validator_message_cites_chunk_optim_path() -> None:
    """Error message must point users at the adapter source file."""
    cfg = _minimal_active_cfg(optimizer="muon")
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "src/axolotl/integrations/protrain/chunk/optim.py" in msg
    # Message should also enumerate the supported set + give a fix.
    assert "adamw_torch" in msg
    assert "remove the ProTrain plugin" in msg


def test_optimizer_validator_is_case_insensitive_accept() -> None:
    """Mixed-case supported names must still be accepted."""
    cfg = _minimal_active_cfg(optimizer="AdamW_Torch")
    ProTrainArgs.model_validate(cfg)


def test_optimizer_validator_skips_when_protrain_inactive() -> None:
    """An unsupported optimizer is fine if ProTrain isn't enabled."""
    cfg = {
        "protrain_auto_memory": False,
        "optimizer": "lion_pytorch",
    }
    ProTrainArgs.model_validate(cfg)


# ---------------------------------------------------------------------
# Mode-force flag mutex (Mode A vs Mode B vs Mode C explicit overrides)
# ---------------------------------------------------------------------


def test_force_replicated_cpu_offload_default_is_none() -> None:
    """Mode B force knob defaults to None (auto-mode picks unless overridden)."""
    args = ProTrainArgs()
    assert args.protrain_force_replicated_cpu_offload is None


def test_force_replicated_cpu_offload_alone_validates() -> None:
    """Mode B knob set in isolation is the supported single-flag path."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_replicated_cpu_offload=True,
    )
    ProTrainArgs.model_validate(cfg)


def test_force_mode_mutex_rejects_modeA_plus_modeB() -> None:
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_all_persistent=True,
        protrain_force_replicated_cpu_offload=True,
    )
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "mutually exclusive" in msg
    assert "protrain_force_all_persistent" in msg
    assert "protrain_force_replicated_cpu_offload" in msg


def test_force_mode_mutex_rejects_modeB_plus_modeC() -> None:
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_replicated_cpu_offload=True,
        protrain_zero3_shard=True,
    )
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "mutually exclusive" in msg
    assert "protrain_force_replicated_cpu_offload" in msg
    assert "protrain_zero3_shard" in msg


def test_force_mode_mutex_rejects_modeA_plus_modeC() -> None:
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_all_persistent=True,
        protrain_zero3_shard=True,
    )
    with pytest.raises(ValidationError) as exc:
        ProTrainArgs.model_validate(cfg)
    msg = str(exc.value)
    assert "mutually exclusive" in msg
    assert "protrain_force_all_persistent" in msg
    assert "protrain_zero3_shard" in msg


def test_force_mode_mutex_allows_modeC_with_modeA_false() -> None:
    """Only one truthy force flag is fine; explicit False on others must not trip the mutex."""
    cfg = _minimal_active_cfg(
        protrain_auto_mode=False,
        protrain_force_all_persistent=False,
        protrain_force_replicated_cpu_offload=False,
        protrain_zero3_shard=True,
    )
    ProTrainArgs.model_validate(cfg)


def test_force_mode_mutex_skips_when_protrain_inactive() -> None:
    """With ProTrain off, even contradictory force flags must not raise."""
    cfg = {
        "protrain_auto_memory": False,
        "protrain_force_all_persistent": True,
        "protrain_zero3_shard": True,
    }
    ProTrainArgs.model_validate(cfg)
