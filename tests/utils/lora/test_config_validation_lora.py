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

    def test_qlora_4bit_validation(self):
        """Legacy ``adapter: qlora`` + ``load_in_4bit: true`` is demoted to
        ``adapter: lora`` + ``load_in_4bit: true`` (QLoRA is just a name for
        that combo). Existing configs keep working unchanged otherwise."""
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
        assert result["adapter"] == "lora"
        assert result["load_in_4bit"] is True

        # Bare ``adapter: qlora`` (no explicit flag) auto-sets load_in_4bit.
        valid_config = DictDefault(
            {
                "adapter": "qlora",
                "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "base_model": "dummy_model",
            }
        )
        result = validate_config(valid_config)
        assert result["adapter"] == "lora"
        assert result["load_in_4bit"] is True

        # ``adapter: qlora`` + ``load_in_8bit`` is contradictory.
        with pytest.raises(ValueError, match="qlora with load_in_8bit is ambiguous"):
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


class TestStructuredQuantizationConfig:
    """Tests for model_quantization_config's structured discriminator."""

    # --- bnb shorthand (replaces adapter: qlora + load_in_4bit) ---

    def test_bnb_nf4_sets_load_in_4bit(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {"bnb": {"weight_dtype": "nf4"}},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        # adapter stays lora; QLoRA-ness is the load_in_4bit flag.
        assert result["adapter"] == "lora"
        assert result["load_in_4bit"] is True

    def test_bnb_int8_stays_lora_8bit(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {"bnb": {"weight_dtype": "int8"}},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["load_in_8bit"] is True

    # --- torchao shorthand ---

    @pytest.mark.parametrize("weight_dtype", ["int4", "nf4", "nvfp4"])
    def test_torchao_4bit_keeps_lora_adapter(self, weight_dtype):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "torchao": {"weight_dtype": weight_dtype}
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        # adapter stays lora; the quantized base lives in model_quantization_config.
        assert result["adapter"] == "lora"
        assert (
            result["model_quantization_config"]["torchao"]["weight_dtype"]
            == weight_dtype
        )

    @pytest.mark.parametrize("weight_dtype", ["int8", "fp8"])
    def test_torchao_weight_only_stays_lora(self, weight_dtype):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "torchao": {"weight_dtype": weight_dtype}
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert not result.get("load_in_4bit")
        assert not result.get("load_in_8bit")

    def test_torchao_mxfp4_passes_schema(self):
        """mxfp4 is rejected at the loader (not the schema) with a pointer
        to quantize_moe_experts."""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {"torchao": {"weight_dtype": "mxfp4"}},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"].torchao.weight_dtype == "mxfp4"

    # --- discriminator validation ---

    def test_discriminator_requires_exactly_one_backend(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="exactly one"):
            validate_config(cfg)

    def test_discriminator_rejects_multiple_backends(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "bnb": {"weight_dtype": "nf4"},
                    "torchao": {"weight_dtype": "int4"},
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="exactly one"):
            validate_config(cfg)

    # --- BC: legacy string form keeps working ---

    def test_legacy_mxfp4config_string_unchanged(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": "Mxfp4Config",
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"] == "Mxfp4Config"

    def test_legacy_pre_existing_qlora_path_demoted(self):
        """Legacy ``adapter: qlora`` + ``load_in_4bit: true`` ends up as
        ``adapter: lora`` + ``load_in_4bit: true`` (functionally identical)."""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["load_in_4bit"] is True

    # --- Conflict surfaces ---

    def test_torchao_with_load_in_4bit_errors(self):
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "model_quantization_config": {"torchao": {"weight_dtype": "int4"}},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"quantize_moe_experts": True}, "quantize_moe_experts: true"),
            ({"gptq": True}, "gptq: true"),
        ],
    )
    def test_torchao_rejects_competing_quant(self, extra, match):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {"torchao": {"weight_dtype": "int4"}},
                **extra,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match=match):
            validate_config(cfg)

    def test_torchao_merge_requires_legacy(self):
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "model_quantization_config": {"torchao": {"weight_dtype": "int4"}},
                "merge_lora": True,
                "merge_method": "memory_efficient",
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="merge_method: legacy"):
            validate_config(cfg)

    def test_torchao_merge_legacy_ok(self):
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "model_quantization_config": {"torchao": {"weight_dtype": "int4"}},
                "merge_lora": True,
                "merge_method": "legacy",
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["merge_method"] == "legacy"

    def test_torchao_with_dora_ok(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {"torchao": {"weight_dtype": "int4"}},
                "peft_use_dora": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["peft_use_dora"] is True
