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
