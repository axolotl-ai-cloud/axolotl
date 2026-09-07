import pytest
import torch.nn as nn
from pydantic import ValidationError

from axolotl.loaders.adapter import load_lora
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.peft import LoraConfig as AxLoraConfig


class TestLoRAConfigValidation:
    """Test suite for LoRA/QLoRA configuration validation"""

    def test_basic_configuration_validation(self):
        """Test basic LoRA configuration validation"""

        valid_config = DictDefault(
            {
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_target_modules": ["q_proj", "v_proj"],
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )

        result = validate_config(valid_config)
        assert result["adapter"] == "lora"

        # DoRA is now compatible with lora kernels
        dora_kernel_config = DictDefault(
            {
                "adapter": "lora",
                "lora_mlp_kernel": True,
                "peft_use_dora": True,
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        result = validate_config(dora_kernel_config)
        assert result["lora_mlp_kernel"] is True
        assert result["peft_use_dora"] is True

    def test_lora_rank_alpha_pattern_passthrough(self):
        """Test that lora_rank_pattern / lora_alpha_pattern survive validation."""
        rank_pattern = {r".*\.visual\.blocks\.\d+\.(attn|mlp)\..*": 128}
        alpha_pattern = {r".*\.visual\.blocks\.\d+\.(attn|mlp)\..*": 128}

        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 64,
                "lora_target_modules": ["q_proj", "v_proj"],
                "lora_rank_pattern": rank_pattern,
                "lora_alpha_pattern": alpha_pattern,
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )

        result = validate_config(cfg)
        assert result["lora_rank_pattern"] == rank_pattern
        assert result["lora_alpha_pattern"] == alpha_pattern

    def test_lora_pattern_empty_dict_validates(self):
        """Empty pattern dicts are accepted and are dropped at the adapter wiring layer."""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_modules": ["q_proj"],
                "lora_rank_pattern": {},
                "lora_alpha_pattern": {},
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        result = validate_config(cfg)
        assert result["lora_rank_pattern"] == {}
        assert result["lora_alpha_pattern"] == {}

    def test_lora_pattern_rejects_non_int_values(self):
        """Pydantic should reject non-int values in rank/alpha pattern dicts."""
        with pytest.raises(ValidationError):
            AxLoraConfig(
                adapter="lora",
                lora_r=8,
                lora_alpha=16,
                lora_rank_pattern={r".*\.q_proj$": "not-an-int"},
            )
        with pytest.raises(ValidationError):
            AxLoraConfig(
                adapter="lora",
                lora_r=8,
                lora_alpha=16,
                lora_alpha_pattern={r".*\.q_proj$": 1.5},
            )

    @pytest.mark.parametrize("bad_value", [0, -1, -42])
    def test_lora_pattern_rejects_non_positive_values(self, bad_value):
        """Pattern values must be > 0; alpha=0 would silently zero-out a module."""
        with pytest.raises(ValidationError):
            AxLoraConfig(
                adapter="lora",
                lora_r=8,
                lora_alpha=16,
                lora_rank_pattern={r".*\.q_proj$": bad_value},
            )
        with pytest.raises(ValidationError):
            AxLoraConfig(
                adapter="lora",
                lora_r=8,
                lora_alpha=16,
                lora_alpha_pattern={r".*\.q_proj$": bad_value},
            )

    def test_lora_patterns_reach_peft_lora_config(self):
        """load_lora(config_only=True) propagates patterns to the constructed PEFT LoraConfig."""
        rank_pattern = {r".*\.layers\.0\..*\.(q_proj|v_proj)$": 32}
        alpha_pattern = {r".*\.layers\.0\..*\.(q_proj|v_proj)$": 64}

        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_modules": ["q_proj", "v_proj"],
                "lora_rank_pattern": rank_pattern,
                "lora_alpha_pattern": alpha_pattern,
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        cfg = validate_config(cfg)

        class _DummyModel(nn.Module):
            pass

        _, peft_config = load_lora(_DummyModel(), cfg, config_only=True)
        assert peft_config.rank_pattern == rank_pattern
        assert peft_config.alpha_pattern == alpha_pattern

    def test_lora_patterns_unset_means_empty_in_peft(self):
        """When patterns are unset they must not appear as None on the PEFT LoraConfig."""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_modules": ["q_proj"],
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        cfg = validate_config(cfg)

        class _DummyModel(nn.Module):
            pass

        _, peft_config = load_lora(_DummyModel(), cfg, config_only=True)
        assert peft_config.rank_pattern == {}
        assert peft_config.alpha_pattern == {}

    def test_qlora_4bit_validation(self):
        """Test QLoRA 4-bit configuration validation"""
        valid_config = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        result = validate_config(valid_config)
        assert result["adapter"] == "qlora"
        assert result["load_in_4bit"] is True

        # Test QLoRA without 4-bit (should fail via PEFT validation)
        with pytest.raises(ValueError, match=r"Require cfg\.load_in_4bit"):
            invalid_config = DictDefault(
                {
                    "adapter": "qlora",
                    "load_in_4bit": False,
                    "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "base_model": "dummy_model",
                }
            )
            validate_config(invalid_config)

        # Test QLoRA with 8-bit (incompatible)
        with pytest.raises(ValueError, match="Can't load qlora in 8bit"):
            invalid_config = DictDefault(
                {
                    "adapter": "qlora",
                    "load_in_8bit": True,
                    "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "base_model": "dummy_model",
                }
            )
            validate_config(invalid_config)

    @pytest.mark.parametrize(
        "kernel_field", ["lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel"]
    )
    def test_lora_kernels_trust_remote_code_incompatible(self, kernel_field):
        """Test that lora kernels are incompatible with trust_remote_code"""
        with pytest.raises(ValueError, match="not compatible with trust_remote_code"):
            invalid_config = DictDefault(
                {
                    "adapter": "lora",
                    kernel_field: True,
                    "trust_remote_code": True,
                    "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "base_model": "dummy_model",
                }
            )
            validate_config(invalid_config)

    def test_lora_kernels_trust_remote_code_false(self):
        """Test that lora kernels work when trust_remote_code is false"""
        # Test with trust_remote_code=False, lora kernels should be allowed
        valid_config = DictDefault(
            {
                "adapter": "lora",
                "lora_mlp_kernel": True,
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "trust_remote_code": False,
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        result = validate_config(valid_config)
        assert result["lora_mlp_kernel"] is True
        assert result["lora_qkv_kernel"] is True
        assert result["lora_o_kernel"] is True

        # Test with trust_remote_code=None (unset), kernels should be allowed
        valid_config = DictDefault(
            {
                "adapter": "lora",
                "lora_qkv_kernel": True,
                "trust_remote_code": None,
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        result = validate_config(valid_config)
        assert result["lora_qkv_kernel"] is True
        assert result["trust_remote_code"] is None
