"""Test for config validation for activation offloading."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.utils.activation_checkpointing import (
    CHECKPOINTING_ARG,
    CHECKPOINTING_KWARGS_ARG,
)
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

LORA = {
    "adapter": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_target_linear": True,
}


@pytest.fixture
def validation_warnings(monkeypatch):
    """Capture the validation module's logger (caplog loses records once any CLI
    entry point reconfigures logging)."""
    import axolotl.utils.schemas.validation as validation_module

    seen: list[str] = []
    monkeypatch.setattr(
        validation_module.LOG, "warning", lambda msg, *a, **k: seen.append(str(msg))
    )
    return seen


def _normalized(cfg):
    with patch(
        "axolotl.utils.config.load_model_config",
        return_value=SimpleNamespace(model_type="llama", to_dict=dict),
    ):
        normalize_config(cfg)
    return cfg


class TestActivationOffloading:
    """
    Test cases for activation offloading schema validation
    """

    def test_offload_all_with_recompute(self, min_base_cfg):
        cfg = (
            DictDefault(activation_checkpointing={"offload": {"all": True}})
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_checkpointing.enabled is True
        assert cfg.activation_checkpointing.offload.all == "cpu"

    def test_hidden_states_offload_defaults_to_non_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(activation_checkpointing={"offload": {"hidden_states": "cpu"}})
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_checkpointing.enabled is True
        assert cfg.activation_checkpointing.reentrant is False
        assert cfg.activation_checkpointing.offload.hidden_states == "cpu"

    def test_hidden_states_offload_accepts_reentrant_alst(self, min_base_cfg):
        cfg = (
            DictDefault(
                activation_checkpointing={
                    "reentrant": True,
                    "offload": {"hidden_states": "cpu"},
                }
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_checkpointing.reentrant is True
        assert cfg.activation_checkpointing.offload.hidden_states == "cpu"

    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_offload_all_without_recompute(
        self, min_base_cfg, validation_warnings, adapter
    ):
        cfg = (
            DictDefault(
                activation_checkpointing={"enabled": False, "offload": {"all": "cpu"}},
                **(LORA if adapter else {}),
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_checkpointing.enabled is False
        assert cfg.activation_checkpointing.offload.all == "cpu"
        assert any("with no recompute" in msg for msg in validation_warnings)

    @pytest.mark.parametrize("rl", ["grpo", "gdpo"])
    def test_grpo_offload_without_recompute(
        self, min_base_cfg, validation_warnings, rl
    ):
        cfg = (
            DictDefault(
                activation_checkpointing={"enabled": False, "offload": {"all": "cpu"}},
                rl=rl,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_checkpointing.offload.all == "cpu"
        assert any("with no recompute" in msg for msg in validation_warnings)

    @pytest.mark.parametrize(
        "trainer",
        [{"rl": "orpo"}, {"process_reward_model": True}],
        ids=["orpo", "prm"],
    )
    def test_offload_without_recompute_rejected_without_offload_context(
        self, min_base_cfg, trainer
    ):
        cfg = (
            DictDefault(
                activation_checkpointing={"enabled": False, "offload": {"all": "cpu"}},
                **trainer,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="without recompute is not supported"):
            validate_config(cfg)

    def test_hidden_states_offload_requires_checkpointing(self, min_base_cfg):
        cfg = (
            DictDefault(
                activation_checkpointing={
                    "enabled": False,
                    "offload": {"hidden_states": "cpu"},
                }
            )
            | min_base_cfg
        )

        with pytest.raises(
            ValueError,
            match=r"`activation_checkpointing.offload.hidden_states` requires "
            "checkpointing",
        ):
            validate_config(cfg)

    @pytest.mark.parametrize("target", ["hidden_states", "saves", "all"])
    def test_disk_offloading_is_disabled(self, min_base_cfg, target):
        block = {"offload": {target: "disk"}}
        if target == "saves":
            block["selective"] = True
        cfg = DictDefault(activation_checkpointing=block) | min_base_cfg

        with pytest.raises(
            ValueError,
            match=rf"offload.{target}: disk offload is disabled pending a fix",
        ):
            validate_config(cfg)

    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_offload_all_with_recompute_does_not_warn(
        self, min_base_cfg, validation_warnings, adapter
    ):
        cfg = (
            DictDefault(
                activation_checkpointing={"offload": {"all": "cpu"}},
                **(LORA if adapter else {}),
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_checkpointing.enabled is True
        assert not any("with no recompute" in msg for msg in validation_warnings)

    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_offload_all_defaults_to_non_reentrant(self, min_base_cfg, adapter):
        cfg = (
            DictDefault(
                activation_checkpointing={"offload": {"all": "cpu"}},
                **(LORA if adapter else {}),
            )
            | min_base_cfg
        )

        cfg = _normalized(validate_config(cfg))
        assert cfg.activation_checkpointing.reentrant is False

    def test_offload_all_keeps_explicit_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                activation_checkpointing={"reentrant": True, "offload": {"all": "cpu"}},
                **LORA,
            )
            | min_base_cfg
        )

        cfg = _normalized(validate_config(cfg))
        assert cfg.activation_checkpointing.reentrant is True

    def test_offload_all_leaves_zero3_reentrant_default(self, min_base_cfg, tmp_path):
        ds_config = tmp_path / "ds.json"
        ds_config.write_text('{"zero_optimization": {"stage": 3}}')
        cfg = (
            DictDefault(
                activation_checkpointing={"offload": {"all": "cpu"}},
                deepspeed=str(ds_config),
                **LORA,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert "reentrant" not in cfg.activation_checkpointing

        # normalize_config then applies the reentrant default, which the trainer
        # builder passes to the Trainer.
        _normalized(cfg)
        args: dict = {}
        TrainerBuilderBase._configure_gradient_checkpointing(
            SimpleNamespace(cfg=cfg), args
        )
        assert args[CHECKPOINTING_ARG] is True
        assert args[CHECKPOINTING_KWARGS_ARG] == {"use_reentrant": True}
        assert args["activation_offloading"] is True
