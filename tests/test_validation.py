"""Module for testing the validation module"""

import logging
import os
import unittest
from typing import Optional

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.wandb_ import setup_wandb_env_vars


class ValidationTest(unittest.TestCase):
    """
    Test the validation module
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test_load_4bit_deprecate(self):
        cfg = DictDefault(
            {
                "load_4bit": True,
            }
        )

        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_batch_size_unused_warning(self):
        cfg = DictDefault(
            {
                "batch_size": 32,
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert "batch_size is not recommended" in self._caplog.records[0].message

    def test_qlora(self):
        base_cfg = DictDefault(
            {
                "adapter": "qlora",
            }
        )

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "load_in_8bit": True,
            }
        )

        with pytest.raises(ValueError, match=r".*8bit.*"):
            validate_config(cfg)

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "gptq": True,
            }
        )

        with pytest.raises(ValueError, match=r".*gptq.*"):
            validate_config(cfg)

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "load_in_4bit": False,
            }
        )

        with pytest.raises(ValueError, match=r".*4bit.*"):
            validate_config(cfg)

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "load_in_4bit": True,
            }
        )

        validate_config(cfg)

    def test_qlora_merge(self):
        base_cfg = DictDefault(
            {
                "adapter": "qlora",
                "merge_lora": True,
            }
        )

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "load_in_8bit": True,
            }
        )

        with pytest.raises(ValueError, match=r".*8bit.*"):
            validate_config(cfg)

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "gptq": True,
            }
        )

        with pytest.raises(ValueError, match=r".*gptq.*"):
            validate_config(cfg)

        cfg = base_cfg | DictDefault(  # pylint: disable=unsupported-binary-operation
            {
                "load_in_4bit": True,
            }
        )

        with pytest.raises(ValueError, match=r".*4bit.*"):
            validate_config(cfg)

    def test_hf_use_auth_token(self):
        cfg = DictDefault(
            {
                "push_dataset_to_hub": "namespace/repo",
            }
        )

        with pytest.raises(ValueError, match=r".*hf_use_auth_token.*"):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "push_dataset_to_hub": "namespace/repo",
                "hf_use_auth_token": True,
            }
        )
        validate_config(cfg)

    def test_gradient_accumulations_or_batch_size(self):
        cfg = DictDefault(
            {
                "gradient_accumulation_steps": 1,
                "batch_size": 1,
            }
        )

        with pytest.raises(
            ValueError, match=r".*gradient_accumulation_steps or batch_size.*"
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "batch_size": 1,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "gradient_accumulation_steps": 1,
            }
        )

        validate_config(cfg)

    def test_falcon_fsdp(self):
        regex_exp = r".*FSDP is not supported for falcon models.*"

        # Check for lower-case
        cfg = DictDefault(
            {
                "base_model": "tiiuae/falcon-7b",
                "fsdp": ["full_shard", "auto_wrap"],
            }
        )

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

        # Check for upper-case
        cfg = DictDefault(
            {
                "base_model": "Falcon-7b",
                "fsdp": ["full_shard", "auto_wrap"],
            }
        )

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "base_model": "tiiuae/falcon-7b",
            }
        )

        validate_config(cfg)

    def test_mpt_gradient_checkpointing(self):
        regex_exp = r".*gradient_checkpointing is not supported for MPT models*"

        # Check for lower-case
        cfg = DictDefault(
            {
                "base_model": "mosaicml/mpt-7b",
                "gradient_checkpointing": True,
            }
        )

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

    def test_flash_optimum(self):
        cfg = DictDefault(
            {
                "flash_optimum": True,
                "adapter": "lora",
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "BetterTransformers probably doesn't work with PEFT adapters"
                in record.message
                for record in self._caplog.records
            )

        cfg = DictDefault(
            {
                "flash_optimum": True,
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "probably set bfloat16 or float16" in record.message
                for record in self._caplog.records
            )

        cfg = DictDefault(
            {
                "flash_optimum": True,
                "fp16": True,
            }
        )
        regex_exp = r".*AMP is not supported.*"

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "flash_optimum": True,
                "bf16": True,
            }
        )
        regex_exp = r".*AMP is not supported.*"

        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

    def test_adamw_hyperparams(self):
        cfg = DictDefault(
            {
                "optimizer": None,
                "adam_epsilon": 0.0001,
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "adamw hyperparameters found, but no adamw optimizer set"
                in record.message
                for record in self._caplog.records
            )

        cfg = DictDefault(
            {
                "optimizer": "adafactor",
                "adam_beta1": 0.0001,
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "adamw hyperparameters found, but no adamw optimizer set"
                in record.message
                for record in self._caplog.records
            )

        cfg = DictDefault(
            {
                "optimizer": "adamw_bnb_8bit",
                "adam_beta1": 0.9,
                "adam_beta2": 0.99,
                "adam_epsilon": 0.0001,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "optimizer": "adafactor",
            }
        )

        validate_config(cfg)

    def test_packing(self):
        cfg = DictDefault(
            {
                "max_packed_sequence_len": 2048,
            }
        )
        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "max_packed_sequence_len will be deprecated in favor of sample_packing"
                in record.message
                for record in self._caplog.records
            )

        cfg = DictDefault(
            {
                "sample_packing": True,
                "pad_to_sequence_len": None,
            }
        )
        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "`pad_to_sequence_len: true` is recommended when using sample_packing"
                in record.message
                for record in self._caplog.records
            )

        cfg = DictDefault(
            {
                "max_packed_sequence_len": 2048,
                "sample_packing": True,
            }
        )
        regex_exp = r".*set only one of max_packed_sequence_len \(deprecated soon\) or sample_packing.*"
        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)

    def test_merge_lora_no_bf16_fail(self):
        """
        This is assumed to be run on a CPU machine, so bf16 is not supported.
        """

        cfg = DictDefault(
            {
                "bf16": True,
            }
        )

        with pytest.raises(ValueError, match=r".*AMP is not supported on this GPU*"):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "bf16": True,
                "merge_lora": True,
            }
        )

        validate_config(cfg)

    def test_sharegpt_deprecation(self):
        cfg = DictDefault(
            {"datasets": [{"path": "lorem/ipsum", "type": "sharegpt:chat"}]}
        )
        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "`type: sharegpt:chat` will soon be deprecated." in record.message
                for record in self._caplog.records
            )
        assert cfg.datasets[0].type == "sharegpt"

        cfg = DictDefault(
            {"datasets": [{"path": "lorem/ipsum", "type": "sharegpt_simple:load_role"}]}
        )
        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "`type: sharegpt_simple` will soon be deprecated." in record.message
                for record in self._caplog.records
            )
        assert cfg.datasets[0].type == "sharegpt:load_role"

    def test_no_conflict_save_strategy(self):
        cfg = DictDefault(
            {
                "save_strategy": "epoch",
                "save_steps": 10,
            }
        )

        with pytest.raises(
            ValueError, match=r".*save_strategy and save_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "save_strategy": "no",
                "save_steps": 10,
            }
        )

        with pytest.raises(
            ValueError, match=r".*save_strategy and save_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "save_strategy": "steps",
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "save_strategy": "steps",
                "save_steps": 10,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "save_steps": 10,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "save_strategy": "no",
            }
        )

        validate_config(cfg)

    def test_no_conflict_eval_strategy(self):
        cfg = DictDefault(
            {
                "evaluation_strategy": "epoch",
                "eval_steps": 10,
            }
        )

        with pytest.raises(
            ValueError, match=r".*evaluation_strategy and eval_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "evaluation_strategy": "no",
                "eval_steps": 10,
            }
        )

        with pytest.raises(
            ValueError, match=r".*evaluation_strategy and eval_steps mismatch.*"
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "evaluation_strategy": "steps",
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "evaluation_strategy": "steps",
                "eval_steps": 10,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "eval_steps": 10,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "evaluation_strategy": "no",
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "evaluation_strategy": "epoch",
                "val_set_size": 0,
            }
        )

        with pytest.raises(
            ValueError,
            match=r".*eval_steps and evaluation_strategy are not supported with val_set_size == 0.*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "eval_steps": 10,
                "val_set_size": 0,
            }
        )

        with pytest.raises(
            ValueError,
            match=r".*eval_steps and evaluation_strategy are not supported with val_set_size == 0.*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "val_set_size": 0,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "eval_steps": 10,
                "val_set_size": 0.01,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "evaluation_strategy": "epoch",
                "val_set_size": 0.01,
            }
        )

        validate_config(cfg)

    def test_eval_table_size_conflict_eval_packing(self):
        cfg = DictDefault(
            {
                "sample_packing": True,
                "eval_table_size": 100,
            }
        )

        with pytest.raises(
            ValueError, match=r".*Please set 'eval_sample_packing' to false.*"
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "sample_packing": True,
                "eval_sample_packing": False,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "sample_packing": False,
                "eval_table_size": 100,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "sample_packing": True,
                "eval_table_size": 100,
                "eval_sample_packing": False,
            }
        )

        validate_config(cfg)

    def test_load_in_x_bit_without_adapter(self):
        cfg = DictDefault(
            {
                "load_in_4bit": True,
            }
        )

        with pytest.raises(
            ValueError,
            match=r".*load_in_8bit and load_in_4bit are not supported without setting an adapter.*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "load_in_8bit": True,
            }
        )

        with pytest.raises(
            ValueError,
            match=r".*load_in_8bit and load_in_4bit are not supported without setting an adapter.*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "load_in_4bit": True,
                "adapter": "qlora",
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "load_in_8bit": True,
                "adapter": "lora",
            }
        )

        validate_config(cfg)

    def test_warmup_step_no_conflict(self):
        cfg = DictDefault(
            {
                "warmup_steps": 10,
                "warmup_ratio": 0.1,
            }
        )

        with pytest.raises(
            ValueError,
            match=r".*warmup_steps and warmup_ratio are mutually exclusive*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "warmup_steps": 10,
            }
        )

        validate_config(cfg)

        cfg = DictDefault(
            {
                "warmup_ratio": 0.1,
            }
        )

        validate_config(cfg)

    def test_add_tokens_adapter(self):
        cfg = DictDefault(
            {"adapter": "qlora", "load_in_4bit": True, "tokens": ["<|imstart|>"]}
        )

        with pytest.raises(
            ValueError,
            match=r".*lora_modules_to_save not properly set yet adding new tokens*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "tokens": ["<|imstart|>"],
                "lora_modules_to_save": ["embed_tokens"],
            }
        )

        with pytest.raises(
            ValueError,
            match=r".*lora_modules_to_save not properly set yet adding new tokens*",
        ):
            validate_config(cfg)

        cfg = DictDefault(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "tokens": ["<|imstart|>"],
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
            }
        )

        validate_config(cfg)


class ValidationWandbTest(ValidationTest):
    """
    Validation test for wandb
    """

    def test_wandb_set_run_id_to_name(self):
        cfg = DictDefault(
            {
                "wandb_run_id": "foo",
            }
        )

        with self._caplog.at_level(logging.WARNING):
            validate_config(cfg)
            assert any(
                "wandb_run_id sets the ID of the run. If you would like to set the name, please use wandb_name instead."
                in record.message
                for record in self._caplog.records
            )

            assert cfg.wandb_name == "foo" and cfg.wandb_run_id == "foo"

        cfg = DictDefault(
            {
                "wandb_name": "foo",
            }
        )

        validate_config(cfg)

        assert cfg.wandb_name == "foo" and cfg.wandb_run_id is None

    def test_wandb_sets_env(self):
        cfg = DictDefault(
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

        validate_config(cfg)

        setup_wandb_env_vars(cfg)

        assert os.environ.get("WANDB_PROJECT", "") == "foo"
        assert os.environ.get("WANDB_NAME", "") == "bar"
        assert os.environ.get("WANDB_RUN_ID", "") == "bat"
        assert os.environ.get("WANDB_ENTITY", "") == "baz"
        assert os.environ.get("WANDB_MODE", "") == "online"
        assert os.environ.get("WANDB_WATCH", "") == "false"
        assert os.environ.get("WANDB_LOG_MODEL", "") == "checkpoint"
        assert os.environ.get("WANDB_DISABLED", "") != "true"

    def test_wandb_set_disabled(self):
        cfg = DictDefault({})

        validate_config(cfg)

        setup_wandb_env_vars(cfg)

        assert os.environ.get("WANDB_DISABLED", "") == "true"

        cfg = DictDefault(
            {
                "wandb_project": "foo",
            }
        )

        validate_config(cfg)

        setup_wandb_env_vars(cfg)

        assert os.environ.get("WANDB_DISABLED", "") != "true"
