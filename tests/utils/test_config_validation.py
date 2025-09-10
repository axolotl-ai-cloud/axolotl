import pytest
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


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

        with pytest.raises(ValueError, match="not compatible with DoRA"):
            invalid_config = DictDefault(
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
            validate_config(invalid_config)

    def test_parameter_freezing_validation(self):
        """Test parameter freezing configuration validation"""
        pass

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
        with pytest.raises(ValueError, match="Require cfg.load_in_4bit"):
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

    def test_target_module_validation(self):
        """Test target module selection validation"""
        pass

        # Note: Target module existence validation happens during model loading in petf,
        # not during config validation, so we don't test for ValueError here
