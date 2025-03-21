# pylint: disable=too-many-lines
"""Module for testing the validation module"""

import logging
import os
import warnings
from typing import Optional

import pytest
from pydantic import ValidationError

from axolotl.utils import is_comet_available
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.mlflow_ import setup_mlflow_env_vars
from axolotl.utils.models import check_model_config
from axolotl.utils.schemas.config import AxolotlConfigWCapabilities
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

    def test_defaults(self, minimal_cfg):
        test_cfg = DictDefault(
            {
                "weight_decay": None,
            }
            | minimal_cfg
        )
        cfg = validate_config(test_cfg)

        assert cfg.train_on_inputs is False
        assert cfg.weight_decay is None

    def test_zero3_qlora_use_reentrant_false(self, minimal_cfg):
        test_cfg = DictDefault(
            {
                "deepspeed": "deepspeed_configs/zero3_bf16.json",
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "load_in_4bit": True,
                "adapter": "qlora",
            }
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(test_cfg)
            assert (
                "qlora + zero3 with use_reentrant: false may result in a CheckpointError about recomputed values"
                in self._caplog.records[0].message
            )

    def test_deepspeed_empty(self, minimal_cfg):
        test_cfg = DictDefault(
            {
                "deepspeed": "",
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "load_in_4bit": True,
                "adapter": "qlora",
            }
            | minimal_cfg
        )

        _ = validate_config(test_cfg)

    def test_deepspeed_not_set(self, minimal_cfg):
        test_cfg = DictDefault(
            {
                "deepspeed": None,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "load_in_4bit": True,
                "adapter": "qlora",
            }
            | minimal_cfg
        )

        _ = validate_config(test_cfg)

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
                    "flash_attention": True,
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
                    "env_capabilities": {
                        "torch_version": "2.5.1",
                    },
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
                    "eval_strategy": "epoch",
                    "eval_steps": 10,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*eval_strategy and eval_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_strategy": "no",
                    "eval_steps": 10,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError, match=r".*eval_strategy and eval_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_strategy": "steps",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_strategy": "steps",
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
                    "eval_strategy": "no",
                }
            )
            | minimal_cfg
        )

        validate_config(cfg)

        cfg = (
            DictDefault(
                {
                    "eval_strategy": "epoch",
                    "val_set_size": 0,
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*eval_steps and eval_strategy are not supported with val_set_size == 0.*",
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
            match=r".*eval_steps and eval_strategy are not supported with val_set_size == 0.*",
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
                    "eval_strategy": "epoch",
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
                    "flash_attention": True,
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
                    "flash_attention": True,
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
                    "flash_attention": True,
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
                    "flash_attention": True,
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

    def test_hub_model_id_save_value_warns_save_stragey_no(self, minimal_cfg):
        cfg = DictDefault({"hub_model_id": "test", "save_strategy": "no"}) | minimal_cfg

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 1

    def test_hub_model_id_save_value_warns_random_value(self, minimal_cfg):
        cfg = (
            DictDefault({"hub_model_id": "test", "save_strategy": "test"}) | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 1

    def test_hub_model_id_save_value_steps(self, minimal_cfg):
        cfg = (
            DictDefault({"hub_model_id": "test", "save_strategy": "steps"})
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 0

    def test_hub_model_id_save_value_epochs(self, minimal_cfg):
        cfg = (
            DictDefault({"hub_model_id": "test", "save_strategy": "epoch"})
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 0

    def test_hub_model_id_save_value_none(self, minimal_cfg):
        cfg = DictDefault({"hub_model_id": "test", "save_strategy": None}) | minimal_cfg

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 0

    def test_hub_model_id_save_value_no_set_save_strategy(self, minimal_cfg):
        cfg = DictDefault({"hub_model_id": "test"}) | minimal_cfg

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert len(self._caplog.records) == 0

    def test_dpo_beta_deprecation(self, minimal_cfg):
        cfg = DictDefault({"dpo_beta": 0.2}) | minimal_cfg

        with self._caplog.at_level(logging.WARNING):
            new_cfg = validate_config(cfg)
            assert new_cfg["rl_beta"] == 0.2
            assert new_cfg["dpo_beta"] is None
            assert len(self._caplog.records) == 1

    def test_eval_strategy_remap(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "evaluation_strategy": "steps",
                }
            )
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            new_cfg = validate_config(cfg)
            assert new_cfg.eval_strategy == "steps"
            assert (
                "evaluation_strategy is deprecated, use eval_strategy instead"
                in self._caplog.records[0].message
            )

    def test_torch_version_adopt_req(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "optimizer": "adopt_adamw",
                }
            )
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*ADOPT optimizer is incompatible with torch version*",
        ):
            env_capabilities = {"torch_version": "2.3.0"}
            capabilities = {"bf16": False}
            _ = validate_config(
                cfg, capabilities=capabilities, env_capabilities=env_capabilities
            )

        env_capabilities = {"torch_version": "2.5.1"}
        capabilities = {"bf16": False}
        _ = validate_config(
            cfg, capabilities=capabilities, env_capabilities=env_capabilities
        )

        env_capabilities = {"torch_version": "2.5.2"}
        capabilities = {"bf16": False}
        _ = validate_config(
            cfg, capabilities=capabilities, env_capabilities=env_capabilities
        )


class TestTorchCompileValidation(BaseValidation):
    """
    test suite for when torch_compile is set to 'auto'
    """

    def test_torch_compile_auto(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "torch_compile": "auto",
                }
            )
            | minimal_cfg
        )

        env_capabilities = {"torch_version": "2.5.1"}
        capabilities = {"bf16": True}
        updated_cfg = validate_config(
            cfg, capabilities=capabilities, env_capabilities=env_capabilities
        )

        assert updated_cfg.torch_compile is True

        env_capabilities = {"torch_version": "2.4.1"}
        capabilities = {"bf16": True}
        updated_cfg = validate_config(
            cfg, capabilities=capabilities, env_capabilities=env_capabilities
        )

        assert updated_cfg.torch_compile is False

        env_capabilities = {}
        capabilities = {"bf16": True}
        updated_cfg = validate_config(
            cfg, capabilities=capabilities, env_capabilities=env_capabilities
        )

        assert updated_cfg.torch_compile is False


class TestSampleOptimConfigValidation(BaseValidation):
    """
    test configurations for sample optimizations like batch flattening
    """

    def test_batch_flattening_auto_enables(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "flash_attention": True,
                    "sample_packing": None,
                    "micro_batch_size": 2,
                    "batch_flattening": "auto",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg["batch_flattening"] is True

    def test_batch_flattening_auto_no_fa(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "flash_attention": False,
                    "sample_packing": None,
                    "micro_batch_size": 2,
                    "batch_flattening": "auto",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg["batch_flattening"] is False

    def test_batch_flattening_auto_mbsz_1(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "flash_attention": True,
                    "sample_packing": None,
                    "micro_batch_size": 1,
                    "batch_flattening": "auto",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg["batch_flattening"] is False

    def test_batch_flattening_auto_packing(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "flash_attention": True,
                    "sample_packing": True,
                    "micro_batch_size": 2,
                    "batch_flattening": "auto",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)
        assert new_cfg["batch_flattening"] is False


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


@pytest.mark.skipif(is_comet_available() is False, reason="comet_ml is not installed")
class TestValidationComet(BaseValidation):
    """
    Validation test for comet
    """

    def test_comet_sets_env(self, minimal_cfg):
        from axolotl.utils.comet_ import setup_comet_env_vars

        comet_config = {
            "comet_api_key": "foo",
            "comet_workspace": "some_workspace",
            "comet_project_name": "some_project",
            "comet_experiment_key": "some_experiment_key",
            "comet_mode": "get_or_create",
            "comet_online": False,
            "comet_experiment_config": {
                "auto_histogram_activation_logging": False,
                "auto_histogram_epoch_rate": 2,
                "auto_histogram_gradient_logging": True,
                "auto_histogram_tensorboard_logging": False,
                "auto_histogram_weight_logging": True,
                "auto_log_co2": False,
                "auto_metric_logging": True,
                "auto_metric_step_rate": 15,
                "auto_output_logging": False,
                "auto_param_logging": True,
                "comet_disabled": False,
                "display_summary_level": 2,
                "distributed_node_identifier": "some_distributed_node_identifier",
                "log_code": True,
                "log_env_cpu": False,
                "log_env_details": True,
                "log_env_disk": False,
                "log_env_gpu": True,
                "log_env_host": False,
                "log_env_network": True,
                "log_git_metadata": False,
                "log_git_patch": True,
                "log_graph": False,
                "name": "some_name",
                "offline_directory": "some_offline_directory",
                "parse_args": True,
                "tags": ["tag1", "tag2"],
            },
        }

        cfg = DictDefault(comet_config) | minimal_cfg

        new_cfg = validate_config(cfg)

        setup_comet_env_vars(new_cfg)

        comet_env = {
            key: value for key, value in os.environ.items() if key.startswith("COMET_")
        }

        assert (
            len(comet_env)
            == len(comet_config) + len(comet_config["comet_experiment_config"]) - 1
        )

        assert comet_env == {
            "COMET_API_KEY": "foo",
            "COMET_AUTO_LOG_CLI_ARGUMENTS": "true",
            "COMET_AUTO_LOG_CO2": "false",
            "COMET_AUTO_LOG_CODE": "true",
            "COMET_AUTO_LOG_DISABLE": "false",
            "COMET_AUTO_LOG_ENV_CPU": "false",
            "COMET_AUTO_LOG_ENV_DETAILS": "true",
            "COMET_AUTO_LOG_ENV_DISK": "false",
            "COMET_AUTO_LOG_ENV_GPU": "true",
            "COMET_AUTO_LOG_ENV_HOST": "false",
            "COMET_AUTO_LOG_ENV_NETWORK": "true",
            "COMET_AUTO_LOG_GIT_METADATA": "false",
            "COMET_AUTO_LOG_GIT_PATCH": "true",
            "COMET_AUTO_LOG_GRAPH": "false",
            "COMET_AUTO_LOG_HISTOGRAM_ACTIVATIONS": "false",
            "COMET_AUTO_LOG_HISTOGRAM_EPOCH_RATE": "2",
            "COMET_AUTO_LOG_HISTOGRAM_GRADIENTS": "true",
            "COMET_AUTO_LOG_HISTOGRAM_TENSORBOARD": "false",
            "COMET_AUTO_LOG_HISTOGRAM_WEIGHTS": "true",
            "COMET_AUTO_LOG_METRIC_STEP_RATE": "15",
            "COMET_AUTO_LOG_METRICS": "true",
            "COMET_AUTO_LOG_OUTPUT_LOGGER": "false",
            "COMET_AUTO_LOG_PARAMETERS": "true",
            "COMET_DISPLAY_SUMMARY_LEVEL": "2",
            "COMET_DISTRIBUTED_NODE_IDENTIFIER": "some_distributed_node_identifier",
            "COMET_EXPERIMENT_KEY": "some_experiment_key",
            "COMET_OFFLINE_DIRECTORY": "some_offline_directory",
            "COMET_PROJECT_NAME": "some_project",
            "COMET_START_EXPERIMENT_NAME": "some_name",
            "COMET_START_EXPERIMENT_TAGS": "tag1,tag2",
            "COMET_START_MODE": "get_or_create",
            "COMET_START_ONLINE": "false",
            "COMET_WORKSPACE": "some_workspace",
        }

        for key in comet_env.keys():
            os.environ.pop(key, None)


class TestValidationMLflow(BaseValidation):
    """
    Validation test for MLflow
    """

    def test_hf_mlflow_artifacts_config_sets_env(self, minimal_cfg):
        cfg = (
            DictDefault(
                {
                    "hf_mlflow_log_artifacts": True,
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)

        assert new_cfg.hf_mlflow_log_artifacts is True

        # Check it's not already present in env
        assert "HF_MLFLOW_LOG_ARTIFACTS" not in os.environ

        setup_mlflow_env_vars(new_cfg)

        assert os.environ.get("HF_MLFLOW_LOG_ARTIFACTS") == "true"

        os.environ.pop("HF_MLFLOW_LOG_ARTIFACTS", None)

    def test_mlflow_not_used_by_default(self, minimal_cfg):
        cfg = DictDefault({}) | minimal_cfg

        new_cfg = validate_config(cfg)

        setup_mlflow_env_vars(new_cfg)

        assert cfg.use_mlflow is not True

        cfg = (
            DictDefault(
                {
                    "mlflow_experiment_name": "foo",
                }
            )
            | minimal_cfg
        )

        new_cfg = validate_config(cfg)

        setup_mlflow_env_vars(new_cfg)

        assert new_cfg.use_mlflow is True

        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
