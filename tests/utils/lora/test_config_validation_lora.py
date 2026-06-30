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
    """Tests for model_quantization_config's backend-discriminated union."""

    # --- bnb shorthand (replaces adapter: qlora + load_in_4bit) ---

    def test_bnb_nf4_sets_load_in_4bit(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        # adapter stays lora; QLoRA-ness is the load_in_4bit flag.
        assert result["adapter"] == "lora"
        assert result["load_in_4bit"] is True
        assert result["load_in_8bit"] is False

    def test_bnb_int8_stays_lora_8bit(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "int8",
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["load_in_8bit"] is True
        assert result["load_in_4bit"] is False

    # --- torchao shorthand ---

    @pytest.mark.parametrize("weight_dtype", ["int4", "nf4", "nvfp4"])
    def test_torchao_4bit_keeps_lora_adapter(self, weight_dtype):
        # int4 has no autograd support in torchao; training requires the
        # LoRA kernels' explicit dequant fwd/bwd.
        kernels = (
            {"lora_mlp_kernel": True, "lora_qkv_kernel": True, "lora_o_kernel": True}
            if weight_dtype == "int4"
            else {}
        )
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": weight_dtype,
                },
                **kernels,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        # adapter stays lora; the quantized base lives in model_quantization_config.
        assert result["adapter"] == "lora"
        assert result["model_quantization_config"]["weight_dtype"] == weight_dtype

    def test_torchao_int4_requires_lora_kernels(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="requires the LoRA kernels"):
            validate_config(cfg)

    @pytest.mark.parametrize("weight_dtype", ["int8", "fp8"])
    def test_torchao_weight_only_stays_lora(self, weight_dtype):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": weight_dtype,
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert not result.get("load_in_4bit")
        assert not result.get("load_in_8bit")

    # --- structured mxfp4 / fp8 backends ---

    def test_mxfp4_backend_with_kwargs(self):
        cfg = DictDefault(
            {
                "model_quantization_config": {
                    "backend": "mxfp4",
                    "config_kwargs": {"dequantize": True},
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"]["backend"] == "mxfp4"
        assert result["model_quantization_config"]["config_kwargs"] == {
            "dequantize": True
        }

    def test_fp8_backend(self):
        cfg = DictDefault(
            {
                "model_quantization_config": {"backend": "fp8"},
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"]["backend"] == "fp8"

    # --- discriminator validation ---

    def test_missing_backend_rejected(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {"weight_dtype": "int4"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_unknown_backend_rejected(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "bitsandbytes",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_typo_sibling_key_rejected(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                    "weight_dtypo": "int8",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    @pytest.mark.parametrize("weight_dtype", ["int4", "nvfp4", "fp8", "mxfp4"])
    def test_bnb_rejects_torchao_only_dtypes(self, weight_dtype):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": weight_dtype,
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_torchao_mxfp4_rejected_at_schema(self):
        """mxfp4 has no weight-only torchao config; the schema rejects it."""
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "mxfp4",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    # --- bnb/torchao base quants require an adapter ---

    @pytest.mark.parametrize(
        "mqc",
        [
            {"backend": "bnb", "weight_dtype": "nf4"},
            {"backend": "torchao", "weight_dtype": "int8"},
        ],
    )
    def test_base_quant_requires_adapter(self, mqc):
        cfg = DictDefault(
            {
                "model_quantization_config": mqc,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="requires `adapter: lora`"):
            validate_config(cfg)

    # --- BC: legacy string form keeps working ---

    def test_legacy_mxfp4config_string_converted(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": "Mxfp4Config",
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"] == {"backend": "mxfp4"}

    def test_legacy_string_kwargs_folded_into_structured_form(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": "FineGrainedFP8Config",
                "model_quantization_config_kwargs": {
                    "modules_to_not_convert": ["lm_head"]
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"] == {
            "backend": "fp8",
            "config_kwargs": {"modules_to_not_convert": ["lm_head"]},
        }
        assert not result["model_quantization_config_kwargs"]

    def test_qlora_alias_with_legacy_string_errors(self):
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "model_quantization_config": "Mxfp4Config",
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="implies a bnb nf4 base quant"):
            validate_config(cfg)

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

    def test_round_trip_revalidates(self):
        """A validated config (structured form + mirrored legacy flags) must
        validate again unchanged."""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        again = validate_config(DictDefault(dict(result)))
        assert again["adapter"] == "lora"
        assert again["load_in_4bit"] is True

    # --- Conflict surfaces ---

    def test_bnb_nf4_with_explicit_4bit_false_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_4bit": False,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="conflicts with the legacy"):
            validate_config(cfg)

    def test_bnb_nf4_with_load_in_8bit_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_8bit": True,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="conflicts with the legacy"):
            validate_config(cfg)

    def test_bnb_int8_with_load_in_4bit_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_4bit": True,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "int8",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="conflicts with the legacy"):
            validate_config(cfg)

    def test_legacy_string_with_load_in_4bit_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_4bit": True,
                "model_quantization_config": "Mxfp4Config",
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="cannot be combined"):
            validate_config(cfg)

    def test_qlora_alias_with_torchao_backend_errors(self):
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int8",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="implies a bnb nf4 base quant"):
            validate_config(cfg)

    def test_torchao_with_load_in_4bit_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_4bit": True,
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
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
            ({"qat": {"weight_dtype": "int8"}}, "qat"),
            ({"ptq": {"weight_dtype": "int8"}}, "ptq"),
            ({"fp8": True}, "fp8"),
            ({"relora": True}, "relora"),
        ],
    )
    def test_torchao_rejects_competing_quant(self, extra, match):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
                **extra,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match=match):
            validate_config(cfg)

    # --- merge-lora flows ---

    def test_merge_lora_in_training_yaml_with_qlora_errors(self):
        """A leftover ``merge_lora: true`` must not silently drop the quant."""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "merge_lora": True,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="Can't merge"):
            validate_config(cfg)

    def test_merge_lora_in_training_yaml_with_structured_bnb_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "merge_lora": True,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="Can't merge"):
            validate_config(cfg)

    def test_merge_lora_qlora_with_gptq_still_errors(self):
        """The merge path must not skip the qlora contradiction checks."""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "gptq": True,
                "merge_lora": True,
                "load_in_4bit": False,
                "load_in_8bit": False,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="gptq"):
            validate_config(cfg)

    def test_merge_lora_legacy_qlora_config_ok(self):
        """The merge CLI forces load_in_4bit/8bit off; a legacy qlora config
        must still validate in that mode."""
        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": False,
                "load_in_8bit": False,
                "merge_lora": True,
                "lora_model_dir": "./out",
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert not result["load_in_4bit"]

    def test_merge_lora_structured_bnb_config_ok(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "load_in_4bit": False,
                "load_in_8bit": False,
                "merge_lora": True,
                "lora_model_dir": "./out",
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        # merge mode never resurrects the legacy flags from the structured form.
        assert not result["load_in_4bit"]

    def test_torchao_merge_requires_legacy(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
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
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
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
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
                "peft_use_dora": True,
                "lora_mlp_kernel": True,
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["adapter"] == "lora"
        assert result["peft_use_dora"] is True

    def test_torchao_nf4_with_dora_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "nf4",
                },
                "peft_use_dora": True,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="peft_use_dora"):
            validate_config(cfg)

    # --- inference mode ---

    def test_inference_quantized_without_adapter_ok(self):
        cfg = DictDefault(
            {
                "load_in_8bit": True,
                "inference": True,
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["load_in_8bit"] is True
        assert result["adapter"] is None

    def test_inference_torchao_int4_without_kernels_ok(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "inference": True,
                "model_quantization_config": {
                    "backend": "torchao",
                    "weight_dtype": "int4",
                },
                **BASE_CFG,
            }
        )
        result = validate_config(cfg)
        assert result["model_quantization_config"]["weight_dtype"] == "int4"


CAPABILITIES = {
    "n_gpu": 1,
    "n_node": 1,
    "gpu_vendor": "nvidia",
    "compute_capability": "sm_90",
}
ENV_CAPABILITIES = {"torch_version": "2.9.0"}


class TestStructuredBnbValidatorOrdering:
    """The flag-reading validators on the capabilities subclass run before the
    base-class normalizer mirrors the structured bnb form into
    load_in_4bit/load_in_8bit; they must understand both spellings."""

    def test_structured_bnb_int8_skips_kernel_auto_enable(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_target_modules": ["q_proj"],
                "lora_r": 8,
                "lora_alpha": 16,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "int8",
                },
                **BASE_CFG,
            }
        )
        result = validate_config(
            cfg, capabilities=CAPABILITIES, env_capabilities=ENV_CAPABILITIES
        )
        assert result["load_in_8bit"] is True
        assert not result["lora_mlp_kernel"]

    def test_quantize_moe_experts_with_structured_bnb_ok(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_target_parameters": ["experts"],
                "lora_r": 8,
                "lora_alpha": 16,
                "quantize_moe_experts": True,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                },
                **BASE_CFG,
            }
        )
        result = validate_config(
            cfg, capabilities=CAPABILITIES, env_capabilities=ENV_CAPABILITIES
        )
        assert result["quantize_moe_experts"] is True
        assert result["load_in_4bit"] is True

    def test_quantize_moe_experts_without_quant_errors(self):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_target_parameters": ["experts"],
                "lora_r": 8,
                "lora_alpha": 16,
                "quantize_moe_experts": True,
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="quantize_moe_experts requires"):
            validate_config(
                cfg, capabilities=CAPABILITIES, env_capabilities=ENV_CAPABILITIES
            )

    @pytest.mark.parametrize("weight_dtype", ["nf4", "int8"])
    def test_qat_with_structured_bnb_errors(self, weight_dtype):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_target_modules": ["q_proj"],
                "lora_r": 8,
                "lora_alpha": 16,
                "model_quantization_config": {
                    "backend": "bnb",
                    "weight_dtype": weight_dtype,
                },
                "qat": {"weight_dtype": "int8"},
                **BASE_CFG,
            }
        )
        with pytest.raises(ValueError, match="QAT and a"):
            validate_config(
                cfg, capabilities=CAPABILITIES, env_capabilities=ENV_CAPABILITIES
            )


class TestTorchAOQuantDTypeParsing:
    def test_from_string_rejects_unknown(self):
        from axolotl.utils.schemas.enums import TorchAOQuantDType

        with pytest.raises(ValueError, match="Invalid torchao dtype"):
            TorchAOQuantDType.from_string("bogus")

    def test_from_string_nf4_has_migration_hint(self):
        from axolotl.utils.schemas.enums import TorchAOQuantDType

        with pytest.raises(ValueError, match="model_quantization_config"):
            TorchAOQuantDType.from_string("nf4")
