"""Test for config validation for activation offloading."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

LORA = {
    "adapter": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_target_linear": True,
}


@pytest.fixture
def validation_logs(monkeypatch):
    """Capture the validation module's logger (caplog loses records once any CLI
    entry point reconfigures logging)."""
    import axolotl.utils.schemas.validation as validation_module

    seen: dict[str, list[str]] = {"info": [], "warning": []}
    for level, messages in seen.items():
        monkeypatch.setattr(
            validation_module.LOG,
            level,
            lambda msg, *a, _messages=messages, **k: _messages.append(str(msg)),
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

    @pytest.mark.parametrize("mode", [True, "disk"])
    @pytest.mark.parametrize("adapter", [False, True], ids=["full", "lora"])
    def test_trl_offload_without_gc(self, min_base_cfg, validation_logs, mode, adapter):
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
        recompute_hint = [
            msg for msg in validation_logs["warning"] if "with no recompute" in msg
        ]
        assert bool(recompute_hint) is not adapter

    @pytest.mark.parametrize("mode", ["legacy", "hidden_states"])
    def test_checkpoint_offload_modes_require_gc(self, min_base_cfg, mode):
        cfg = (
            DictDefault(gradient_checkpointing=False, activation_offloading=mode)
            | min_base_cfg
        )

        with pytest.raises(
            ValueError, match=f"activation_offloading: {mode} requires gradient"
        ):
            validate_config(cfg)

    def test_trl_offload_with_gc_and_adapter_skips_recompute(
        self, min_base_cfg, validation_logs
    ):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                **LORA,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert any(
            "gradient_checkpointing is not applied" in msg
            for msg in validation_logs["info"]
        )
        assert not any("with no recompute" in msg for msg in validation_logs["warning"])
