"""Validation tests for the Q-GaLore optimizer config gates."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestQGaLoreValidation:
    """Pydantic-level checks for q_galore_adamw8bit."""

    def test_adapter_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            optimizer="q_galore_adamw8bit",
            adapter="lora",
            lora_r=8,
            lora_alpha=16,
            lora_target_linear=True,
        )
        with pytest.raises(ValueError, match="incompatible with adapter"):
            validate_config(cfg)

    def test_fsdp1_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            optimizer="q_galore_adamw8bit",
            fsdp_version=1,
            fsdp_config={"reshard_after_forward": True},
        )
        with pytest.raises(ValueError, match="requires FSDP2"):
            validate_config(cfg)

    def test_defaults_filled(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            optimizer="q_galore_adamw8bit",
            bf16=True,
        )
        cfg = validate_config(cfg)
        assert cfg.optim_target_modules == ["attn", "mlp"]
        assert cfg.qgalore_rank == 256
        assert cfg.qgalore_proj_bits == 4
