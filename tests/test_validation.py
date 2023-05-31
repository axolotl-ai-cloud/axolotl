"""Module for testing the validation module"""

import logging
import unittest
from typing import Optional

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.validation import validate_config


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
