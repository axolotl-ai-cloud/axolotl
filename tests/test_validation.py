"""Module for testing the validation module"""

import logging
import unittest
from typing import Optional

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


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
                "max_packed_sequence_len": 2048,
                "sample_packing": True,
            }
        )
        regex_exp = r".*set only one of max_packed_sequence_len \(deprecated soon\) or sample_packing.*"
        with pytest.raises(ValueError, match=regex_exp):
            validate_config(cfg)
