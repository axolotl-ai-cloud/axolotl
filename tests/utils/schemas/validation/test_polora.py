"""Validation tests for the PoLoRA optimizer config gates."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _polora_cfg(min_base_cfg, **overrides):
    return (
        min_base_cfg
        | DictDefault(
            optimizer="polora",
            adapter="lora",
            lora_r=8,
            lora_alpha=16,
            lora_target_linear=True,
        )
        | DictDefault(**overrides)
    )


class TestPoloraValidation:
    """Pydantic-level checks for polora."""

    def test_lora_adapter_accepted(self, min_base_cfg):
        cfg = validate_config(_polora_cfg(min_base_cfg))
        assert cfg.optimizer == "polora"

    def test_missing_adapter_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(optimizer="polora")
        with pytest.raises(ValueError, match="requires adapter: lora or qlora"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"deepspeed": "deepspeed_configs/zero2.json"},
            {"tensor_parallel_size": 2},
        ],
    )
    def test_sharded_backends_rejected(self, min_base_cfg, overrides):
        with pytest.raises(ValueError, match="not compatible with DeepSpeed"):
            validate_config(_polora_cfg(min_base_cfg, **overrides))

    def test_fsdp2_accepted(self, min_base_cfg):
        cfg = _polora_cfg(
            min_base_cfg,
            fsdp_version=2,
            fsdp_config={"reshard_after_forward": True},
        )
        assert validate_config(cfg).optimizer == "polora"

    def test_fsdp1_rejected(self, min_base_cfg):
        cfg = _polora_cfg(
            min_base_cfg,
            fsdp_version=1,
            fsdp_config={"reshard_after_forward": True},
        )
        with pytest.raises(ValueError, match="requires FSDP2"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"lora_modules_to_save": ["embed_tokens", "lm_head"]},
            {"unfrozen_parameters": ["lm_head"]},
            {"peft_use_dora": True},
            {"peft_trainable_token_indices": [128000]},
            {"lisa_step_interval": 20, "lisa_n_layers": 4},
        ],
    )
    def test_untrainable_params_rejected(self, min_base_cfg, overrides):
        with pytest.raises(ValueError, match="would never be trained"):
            validate_config(_polora_cfg(min_base_cfg, **overrides))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"loraplus_lr_ratio": 8},
            {"embedding_lr": 1e-6},
            {"embedding_lr_scale": 0.5},
        ],
    )
    def test_ignored_learning_rates_rejected(self, min_base_cfg, overrides):
        with pytest.raises(ValueError, match="silently ignored"):
            validate_config(_polora_cfg(min_base_cfg, **overrides))

    def test_relora_rejected(self, min_base_cfg):
        cfg = _polora_cfg(min_base_cfg, relora_steps=100, relora_warmup_steps=10)
        with pytest.raises(ValueError, match="relora resets optimizer state"):
            validate_config(cfg)
