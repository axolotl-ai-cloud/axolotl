"""Test for config validation for activation offloading."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

LORA = {
    "adapter": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_target_linear": True,
}
DISK = pytest.param(
    "disk", marks=pytest.mark.skip(reason="activation_offloading: disk is disabled")
)


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


class TestActivationOffloading:
    """
    Test cases for activation offloading schema validation
    """

    def test_gc_converts_offload_wo_lora(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing="offload",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading is True

    def test_ac_offload_impl_noop_wo_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading is True

    def test_hidden_states_offload_defaults_to_non_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading="hidden_states",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is False
        assert cfg.activation_offloading == "hidden_states"

    def test_hidden_states_offload_accepts_reentrant_alst(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                activation_offloading="hidden_states",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is True
        assert cfg.activation_offloading == "hidden_states"

    @pytest.mark.parametrize("mode", [True, DISK])
    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_trl_offload_without_gc(
        self, min_base_cfg, validation_warnings, mode, adapter
    ):
        cfg = (
            DictDefault(
                gradient_checkpointing=False,
                activation_offloading=mode,
                **(LORA if adapter else {}),
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is False
        assert cfg.activation_offloading == mode
        assert any("with no recompute" in msg for msg in validation_warnings)

    @pytest.mark.parametrize("rl", ["grpo", "gdpo"])
    def test_grpo_offload_without_gc(self, min_base_cfg, validation_warnings, rl):
        cfg = (
            DictDefault(gradient_checkpointing=False, activation_offloading=True, rl=rl)
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_offloading is True
        assert any("with no recompute" in msg for msg in validation_warnings)

    @pytest.mark.parametrize(
        "trainer",
        [{"rl": "orpo"}, {"process_reward_model": True}],
        ids=["orpo", "prm"],
    )
    def test_trl_offload_without_gc_rejected_without_offload_context(
        self, min_base_cfg, trainer
    ):
        cfg = (
            DictDefault(
                gradient_checkpointing=False, activation_offloading=True, **trainer
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="without gradient_checkpointing is not"):
            validate_config(cfg)

    def test_hidden_states_offload_requires_gc(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=False, activation_offloading="hidden_states"
            )
            | min_base_cfg
        )

        with pytest.raises(
            ValueError, match="activation_offloading: hidden_states requires gradient"
        ):
            validate_config(cfg)

    @pytest.mark.parametrize("gradient_checkpointing", [True, False])
    def test_legacy_is_an_alias_of_true(
        self, min_base_cfg, validation_warnings, gradient_checkpointing
    ):
        cfg = (
            DictDefault(
                gradient_checkpointing=gradient_checkpointing,
                activation_offloading="legacy",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.activation_offloading is True
        assert any("legacy is deprecated" in msg for msg in validation_warnings)

    @pytest.mark.parametrize(
        "disk",
        [{"activation_offloading": "disk"}, {"gradient_checkpointing": "offload_disk"}],
        ids=["activation_offloading", "gradient_checkpointing"],
    )
    def test_disk_offloading_is_disabled(self, min_base_cfg, disk):
        cfg = DictDefault({"gradient_checkpointing": True, **disk}) | min_base_cfg

        with pytest.raises(ValueError, match="activation_offloading: disk is disabled"):
            validate_config(cfg)

    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_trl_offload_with_gc(self, min_base_cfg, validation_warnings, adapter):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                **(LORA if adapter else {}),
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading is True
        assert not any("with no recompute" in msg for msg in validation_warnings)

    @pytest.mark.parametrize("mode", [True, DISK])
    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_trl_offload_with_gc_defaults_to_non_reentrant(
        self, min_base_cfg, mode, adapter
    ):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=mode,
                **(LORA if adapter else {}),
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is False

    def test_trl_offload_with_gc_keeps_explicit_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                activation_offloading=True,
                **LORA,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is True

    def test_trl_offload_with_gc_leaves_zero3_reentrant_default(
        self, min_base_cfg, tmp_path
    ):
        ds_config = tmp_path / "ds.json"
        ds_config.write_text('{"zero_optimization": {"stage": 3}}')
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                deepspeed=str(ds_config),
                **LORA,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert "use_reentrant" not in (cfg.gradient_checkpointing_kwargs or {})

        # normalize_config then applies the reentrant default, which the trainer
        # builder passes to the Trainer.
        with patch(
            "axolotl.utils.config.load_model_config",
            return_value=SimpleNamespace(model_type="llama", to_dict=dict),
        ):
            normalize_config(cfg)
        args: dict = {}
        TrainerBuilderBase._configure_gradient_checkpointing(
            SimpleNamespace(cfg=cfg), args
        )
        assert args["gradient_checkpointing"] is True
        assert args["gradient_checkpointing_kwargs"] == {"use_reentrant": True}
        assert args["activation_offloading"] is True
