import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

BASE_CFG = {
    "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "base_model": "dummy_model",
}


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


class TestTorchaoQLoRAConfigValidation:
    """Test suite for torchao QLoRA auto-detection and validation"""

    # --- Auto-detection: torchao ---

    @pytest.mark.parametrize("weight_dtype", ["int4", "int8", "nf4"])
    def test_torchao_auto_detect_from_lora(self, weight_dtype):
        """adapter: lora + peft.backend: torchao auto-upgrades to qlora"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "peft": {"backend": "torchao", "weight_dtype": weight_dtype},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "qlora"
        assert result["peft"]["backend"] == "torchao"

    def test_torchao_explicit_qlora(self):
        """adapter: qlora + peft.backend: torchao works directly"""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "peft": {"backend": "torchao", "weight_dtype": "int4"},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "qlora"

    # --- Auto-detection: bnb ---

    def test_bnb_nf4_auto_detect_from_lora(self):
        """adapter: lora + peft.backend: bnb + weight_dtype: nf4 → qlora + load_in_4bit"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "peft": {"backend": "bnb", "weight_dtype": "nf4"},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "qlora"
        assert result["load_in_4bit"] is True

    def test_bnb_int8_auto_detect_from_lora(self):
        """adapter: lora + peft.backend: bnb + weight_dtype: int8 → lora + load_in_8bit"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "peft": {"backend": "bnb", "weight_dtype": "int8"},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["load_in_8bit"] is True

    def test_bnb_nf4_explicit_qlora_auto_sets_load_in_4bit(self):
        """adapter: qlora + peft.backend: bnb + weight_dtype: nf4 auto-sets load_in_4bit"""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "peft": {"backend": "bnb", "weight_dtype": "nf4"},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "qlora"
        assert result["load_in_4bit"] is True

    # --- Backward compat ---

    def test_old_style_qlora_unchanged(self):
        """Old-style adapter: qlora + load_in_4bit: true still works"""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "qlora"
        assert result["load_in_4bit"] is True

    def test_old_style_lora_8bit_unchanged(self):
        """Old-style adapter: lora + load_in_8bit: true still works"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_8bit": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["load_in_8bit"] is True

    def test_plain_lora_unchanged(self):
        """adapter: lora without peft block stays as lora"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"

    # --- Validation errors ---

    def test_torchao_with_load_in_4bit_errors(self):
        """peft.backend: torchao + load_in_4bit is a conflict"""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "peft": {"backend": "torchao", "weight_dtype": "int4"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="load_in_4bit.*bitsandbytes"):
            validate_config(cfg)

    def test_torchao_with_load_in_8bit_errors(self):
        """peft.backend: torchao + load_in_8bit is a conflict"""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_8bit": True,
                "peft": {"backend": "torchao", "weight_dtype": "int4"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="load_in_4bit.*bitsandbytes"):
            validate_config(cfg)

    def test_torchao_without_weight_dtype_errors(self):
        """peft.backend: torchao without weight_dtype errors"""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "peft": {"backend": "torchao"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="peft.weight_dtype is required"):
            validate_config(cfg)

    def test_weight_dtype_without_backend_errors(self):
        """peft.weight_dtype without peft.backend errors"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "peft": {"weight_dtype": "int4"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="peft.backend is required"):
            validate_config(cfg)

    def test_bnb_unsupported_weight_dtype_errors(self):
        """peft.backend: bnb + unsupported weight_dtype errors"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "peft": {"backend": "bnb", "weight_dtype": "int4"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="not supported with bnb"):
            validate_config(cfg)

    # --- Redundant flags don't conflict ---

    def test_bnb_nf4_with_explicit_load_in_4bit(self):
        """peft.backend: bnb + weight_dtype: nf4 + load_in_4bit: true is fine (redundant)"""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_4bit": True,
                "peft": {"backend": "bnb", "weight_dtype": "nf4"},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "qlora"
        assert result["load_in_4bit"] is True
