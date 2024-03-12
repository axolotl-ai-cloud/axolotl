# pylint: disable=too-many-lines
"""Module for testing the validation module"""

import logging
import os
import warnings
from typing import Optional

import pytest
from pydantic import ValidationError

from axolotl.utils.config import validate_config
from axolotl.utils.config.models.input.v0_4_1 import AxolotlConfigWCapabilities
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import check_model_config
from axolotl.utils.wandb_ import setup_wandb_env_vars

warnings.filterwarnings("error")


@pytest.fixture(name="minimal_cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                }
            ],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
        }
    )


class BaseValidation:
    """
    Base validation module to setup the log capture
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog


# pylint: disable=too-many-public-methods
class TestValidation(BaseValidation):
    """
    Test the validation module
    """

    def test_datasets_min_length(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "datasets": [],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        )

        with pytest.raises(
            ValidationError,
            match=r".*List should have at least 1 item after validation*",
        ):
            validate_config(cfg)

    def test_datasets_min_length_empty(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        )

        with pytest.raises(
            ValueError, match=r".*either datasets or pretraining_dataset is required*"
        ):
            validate_config(cfg)

    def test_pretrain_dataset_min_length(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "pretraining_dataset": [],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 100,
            }
        )

        with pytest.raises(
            ValidationError,
            match=r".*List should have at least 1 item after validation*",
        ):
            validate_config(cfg)

    def test_valid_pretrain_dataset(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "pretraining_dataset": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 100,
            }
        )

        validate_config(cfg)

    def test_valid_sft_dataset(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        )

        validate_config(cfg)

    def test_batch_size_unused_warning(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "micro_batch_size": 4,
                "batch_size": 32,
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert "batch_size is not recommended" in self._caplog.records[0].message

    def test_batch_size_more_params(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "batch_size": 32,
            }
        )

        with pytest.raises(ValueError, match=r".*At least two of*"):
            validate_config(cfg)

    def test_lr_as_float(self, minimal_cfg):
        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "learning_rate": "5e-5",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)

        assert new_cfg.learning_rate == 0.00005

    def test_model_config_remap(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "model_config": {"model_type": "mistral"},
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg.overrides_of_model_config["model_type"] == "mistral"

    def test_model_type_remap(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "model_type": "AutoModelForCausalLM",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg.type_of_model == "AutoModelForCausalLM"

    def test_model_revision_remap(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "model_revision": "main",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg.revision_of_model == "main"

    def test_qlora(self, minimal_cfg):
        base_cfg = (
            DictDefault(
                {
                    "adapter": "qlora",
                }
            )
            | minimal_cfg
        )

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "load_in_8bit": True,
                }
            )
            | base_cfg
        )

        with pytest.raises(ValueError, match=r".*8bit.*"):
            validate_config(cfg)

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "gptq": True,
                }
            )
            | base_cfg
        )

        with pytest.raises(ValueError, match=r".*gptq.*"):
            validate_config(cfg)

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "load_in_4bit": False,
                }
            )
            | base_cfg
        )

        with pytest.raises(ValueError, match=r".*4bit.*"):
            validate_config(cfg)

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "load_in_4bit": True,
                }
            )
            | base_cfg
        )

        validate_config(cfg)

    def test_qlora_merge(self, minimal_cfg):
        base_cfg = (
            DictDefault(
                {
                    "adapter": "qlora",
                    "merge_lora": True,
                }
            )
            | minimal_cfg
        )

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "load_in_8bit": True,
                }
            )
            | base_cfg
        )

        with pytest.raises(ValueError, match=r".*8bit.*"):
            validate_config(cfg)

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "gptq": True,
                }
            )
            | base_cfg
        )

        with pytest.raises(ValueError, match=r".*gptq.*"):
            validate_config(cfg)

        cfg = (
            DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "load_in_4bit": True,
                }
            )
            | base_cfg
        )

        with pytest.raises(ValueError, match=r".*4bit.*"):
            validate_config(cfg)

    def test_hf_use_auth_token(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "push_dataset_to_hub": "namespace/repo",
                }
            )
            | minimal_cfg
        )

        with pytest.raises(ValueError, match=r".*hf_use_auth_token.*"):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "push_dataset_to_hub": "namespace/repo",
                    "hf_use_auth_token": True,
                }
            )
            | minimal_cfg
        )
        validate_config(cfg)

    def test_gradient_accumulations_or_batch_size(self):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "gradient_accumulation_steps": 1,
                "batch_size": 1,
            }
        )

        with pytest.raises(
            ValueError, match=r".*gradient_accumulation_steps or batch_size.*"
        ):
            validate_config(cfg)

    def test_falcon_fsdp(self, minimal_cfg):
        regex_exp = r".*FSDP is not supported for falcon models.*"

        # Check for lower-case
        cfg = (
            DictDefault(
                {
                    "base_model": "tiiuae/falcon-7b",
                    "fsdp": ["full_shard", "auto_wrap"],
                }
            )
            | minimal_cfg
        )

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

        # Check for upper-case
        cfg = (
            DictDefault(
                {
                    "base_model": "Falcon-7b",
                    "fsdp": ["full_shard", "auto_wrap"],
                }
            )
            | minimal_cfg
        )

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "base_model": "tiiuae/falcon-7b",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_mpt_gradient_checkpointing(self, minimal_cfg):
        regex_exp = r".*gradient_checkpointing is not supported for MPT models*"

        # Check for lower-case
        cfg = (
            DictDefault(
                {
                    "base_model": "mosaicml/mpt-7b",
                    "gradient_checkpointing": True,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

    def test_flash_optimum(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "flash_optimum": True,
                    "adapter": "lora",
                    "bf16": False,
                }
            )
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "BetterTransformers probably doesn't work with PEFT adapters"
                in record.message
                for record in self._caplog.records
            )

        cfg = (
            DictDefault(
                {
                    "flash_optimum": True,
                    "bf16": False,
                }
            )
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "probably set bfloat16 or float16" in record.message
                for record in self._caplog.records
            )

        cfg = (
            DictDefault(
                {
                    "flash_optimum": True,
                    "fp16": True,
                }
            )
            | minimal_cfg
        )
        regex_exp = r".*AMP is not supported.*"

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "flash_optimum": True,
                    "bf16": True,
                }
            )
            | minimal_cfg
        )
        regex_exp = r".*AMP is not supported.*"

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

    def test_adamw_hyperparams(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "optimizer": None,
                    "adam_epsilon": 0.0001,
                }
            )
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "adamw hyperparameters found, but no adamw optimizer set"
                in record.message
                for record in self._caplog.records
            )

        cfg = (
            DictDefault(
                {
                    "optimizer": "adafactor",
                    "adam_beta1": 0.0001,
                }
            )
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "adamw hyperparameters found, but no adamw optimizer set"
                in record.message
                for record in self._caplog.records
            )

        cfg = (
            DictDefault(
                {
                    "optimizer": "adamw_bnb_8bit",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.99,
                    "adam_epsilon": 0.0001,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "optimizer": "adafactor",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_deprecated_packing(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "max_packed_sequence_len": 1024,
                }
            )
            | minimal_cfg
        )
        with pytest.raises(
            DeprecationWarning,
            match=r"`max_packed_sequence_len` is no longer supported",
        ):
            validate_config(cfg)

    def test_packing(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "sample_packing": True,
                    "pad_to_sequence_len": None,
                }
            )
            | minimal_cfg
        )
        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "`pad_to_sequence_len: true` is recommended when using sample_packing"
                in record.message
                for record in self._caplog.records
            )

    def test_merge_lora_no_bf16_fail(self, minimal_cfg):
        """
        This is assumed to be run on a CPU machine, so bf16 is not supported.
        """

        cfg = (
            DictDefault(
                {
                    "bf16": True,
                    "capabilities": {"bf16": False},
                }
            )
            | minimal_cfg
        )

        with pytest.raises(ValueError, match=r".*AMP is not supported on this GPU*"):
            AxolotlConfigWCapabilities(**cfg.to_dict())

        cfg = (
            DictDefault(
                {
                    "bf16": True,
                    "merge_lora": True,
                    "capabilities": {"bf16": False},
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_sharegpt_deprecation(self, minimal_cfg):
        cfg = (
            DictDefault(
                {"datasets": [{"path": "lorem/ipsum", "type": "sharegpt:chat"}]}
            )
            | minimal_cfg
        )
        with self._caplog.at_level(logging.WARNING):
            new_cfg = validate_config(cfg)
            assert any(
                "`type: sharegpt:chat` will soon be deprecated." in record.message
                for record in self._caplog.records
            )
        assert new_cfg.datasets[0].type == "sharegpt"

        cfg = (
            DictDefault(
                {
                    "datasets": [
                        {"path": "lorem/ipsum", "type": "sharegpt_simple:load_role"}
                    ]
                }
            )
            | minimal_cfg
        )
        with self._caplog.at_level(logging.WARNING):
            new_cfg = validate_config(cfg)
            assert any(
                "`type: sharegpt_simple` will soon be deprecated." in record.message
                for record in self._caplog.records
            )
        assert new_cfg.datasets[0].type == "sharegpt:load_role"

    def test_no_conflict_save_strategy(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "save_strategy": "epoch",
                    "save_steps": 10,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*save_strategy and save_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "save_strategy": "no",
                    "save_steps": 10,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*save_strategy and save_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "save_strategy": "steps",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "save_strategy": "steps",
                    "save_steps": 10,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "save_steps": 10,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "save_strategy": "no",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_no_conflict_eval_strategy(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "epoch",
                    "eval_steps": 10,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*evaluation_strategy and eval_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "no",
                    "eval_steps": 10,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*evaluation_strategy and eval_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "steps",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "steps",
                    "eval_steps": 10,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_steps": 10,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "no",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "epoch",
                    "val_set_size": 0,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*eval_steps and evaluation_strategy are not supported with val_set_size == 0.*",
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_steps": 10,
                    "val_set_size": 0,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*eval_steps and evaluation_strategy are not supported with val_set_size == 0.*",
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "val_set_size": 0,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_steps": 10,
                    "val_set_size": 0.01,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "epoch",
                    "val_set_size": 0.01,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_eval_table_size_conflict_eval_packing(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "sample_packing": True,
                    "eval_table_size": 100,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*Please set 'eval_sample_packing' to false.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "sample_packing": True,
                    "eval_sample_packing": False,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "sample_packing": False,
                    "eval_table_size": 100,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "sample_packing": True,
                    "eval_table_size": 100,
                    "eval_sample_packing": False,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_load_in_x_bit_without_adapter(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "load_in_4bit": True,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*load_in_8bit and load_in_4bit are not supported without setting an adapter.*",
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "load_in_8bit": True,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*load_in_8bit and load_in_4bit are not supported without setting an adapter.*",
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "load_in_4bit": True,
                    "adapter": "qlora",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "load_in_8bit": True,
                    "adapter": "lora",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_warmup_step_no_conflict(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "warmup_steps": 10,
                    "warmup_ratio": 0.1,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*warmup_steps and warmup_ratio are mutually exclusive*",
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "warmup_steps": 10,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "warmup_ratio": 0.1,
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

    def test_unfrozen_parameters_w_peft_layers_to_transform(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "adapter": "lora",
                    "unfrozen_parameters": [
                        "model.layers.2[0-9]+.block_sparse_moe.gate.*"
                    ],
                    "peft_layers_to_transform": [0, 1],
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*can have unexpected behavior*",
        ):
            validate_config(cfg)

    def test_hub_model_id_save_value_warns(self, minimal_cfg):
        cfg = DictDefault({"hub_model_id": "test"}) | minimal_cfg

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert (
                "set without any models being saved" in self._caplog.records[0].message
            )

    def test_hub_model_id_save_value(self, minimal_cfg):
        cfg = DictDefault({"hub_model_id": "test", "saves_per_epoch": 4}) | minimal_cfg

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 0


class TestValidationCheckModelConfig(BaseValidation):
    """
    Test the validation for the config when the model config is available
    """

    def test_llama_add_tokens_adapter(self, minimal_cfg):
        cfg = (
            DictDefault(
                {"adapter": "qlora", "load_in_4bit": True, "tokens": ["<|imstart|>"]}
            )
            | minimal_cfg
        )
        model_config = DictDefault({"model_type": "llama"})

        with pytest.raises(
            ValueError,
            match=r".*`lora_modules_to_save` not properly set when adding new tokens*",
        ):
            check_model_config(cfg, model_config)

        cfg = (
            DictDefault(
                {
                    "adapter": "qlora",
                    "load_in_4bit": True,
                    "tokens": ["<|imstart|>"],
                    "lora_modules_to_save": ["embed_tokens"],
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*`lora_modules_to_save` not properly set when adding new tokens*",
        ):
            check_model_config(cfg, model_config)

        cfg = (
            DictDefault(
                {
                    "adapter": "qlora",
                    "load_in_4bit": True,
                    "tokens": ["<|imstart|>"],
                    "lora_modules_to_save": ["embed_tokens", "lm_head"],
                }
            )
            | minimal_cfg
        )

        check_model_config(cfg, model_config)

    def test_phi_add_tokens_adapter(self, minimal_cfg):
        cfg = (
            DictDefault(
                {"adapter": "qlora", "load_in_4bit": True, "tokens": ["<|imstart|>"]}
            )
            | minimal_cfg
        )
        model_config = DictDefault({"model_type": "phi"})

        with pytest.raises(
            ValueError,
            match=r".*`lora_modules_to_save` not properly set when adding new tokens*",
        ):
            check_model_config(cfg, model_config)

        cfg = (
            DictDefault(
                {
                    "adapter": "qlora",
                    "load_in_4bit": True,
                    "tokens": ["<|imstart|>"],
                    "lora_modules_to_save": ["embd.wte", "lm_head.linear"],
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*`lora_modules_to_save` not properly set when adding new tokens*",
        ):
            check_model_config(cfg, model_config)

        cfg = (
            DictDefault(
                {
                    "adapter": "qlora",
                    "load_in_4bit": True,
                    "tokens": ["<|imstart|>"],
                    "lora_modules_to_save": ["embed_tokens", "lm_head"],
                }
            )
            | minimal_cfg
        )

        check_model_config(cfg, model_config)


class TestValidationWandb(BaseValidation):
    """
    Validation test for wandb
    """

    def test_wandb_set_run_id_to_name(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "wandb_run_id": "foo",
                }
            )
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            new_cfg = validate_config(cfg)
            assert any(
                "wandb_run_id sets the ID of the run. If you would like to set the name, please use wandb_name instead."
                in record.message
                for record in self._caplog.records
            )

            assert new_cfg.wandb_name == "foo" and new_cfg.wandb_run_id == "foo"

        cfg = (
            DictDefault(
                {
                    "wandb_name": "foo",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)

        assert new_cfg.wandb_name == "foo" and new_cfg.wandb_run_id is None

    def test_wandb_sets_env(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "wandb_project": "foo",
                    "wandb_name": "bar",
                    "wandb_run_id": "bat",
                    "wandb_entity": "baz",
                    "wandb_mode": "online",
                    "wandb_watch": "false",
                    "wandb_log_model": "checkpoint",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)

        setup_wandb_env_vars(new_cfg)

        assert os.environ.get("WANDB_PROJECT", "") == "foo"
        assert os.environ.get("WANDB_NAME", "") == "bar"
        assert os.environ.get("WANDB_RUN_ID", "") == "bat"
        assert os.environ.get("WANDB_ENTITY", "") == "baz"
        assert os.environ.get("WANDB_MODE", "") == "online"
        assert os.environ.get("WANDB_WATCH", "") == "false"
        assert os.environ.get("WANDB_LOG_MODEL", "") == "checkpoint"
        assert os.environ.get("WANDB_DISABLED", "") != "true"

        os.environ.pop("WANDB_PROJECT", None)
        os.environ.pop("WANDB_NAME", None)
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_ENTITY", None)
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_WATCH", None)
        os.environ.pop("WANDB_LOG_MODEL", None)
        os.environ.pop("WANDB_DISABLED", None)

    def test_wandb_set_disabled(self, minimal_cfg):
        cfg = DictDefault({}) | minimal_cfg

        new_cfg = validate_config(cfg)

        setup_wandb_env_vars(new_cfg)

        assert os.environ.get("WANDB_DISABLED", "") == "true"

        cfg = (
            DictDefault(
                {
                    "wandb_project": "foo",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)

        setup_wandb_env_vars(new_cfg)

        assert os.environ.get("WANDB_DISABLED", "") != "true"

        os.environ.pop("WANDB_PROJECT", None)
        os.environ.pop("WANDB_DISABLED", None)
