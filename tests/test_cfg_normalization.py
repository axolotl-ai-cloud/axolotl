"""Module for testing the validation module"""

import logging
import unittest
from typing import Optional

import pytest

from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault


class NormalizationTest(unittest.TestCase):
    """
    Test the cfg normalization module
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test_lora_to_peft(self):
        base_cfg = DictDefault(
            {
                "gradient_accumulation_steps": 1,
                "micro_batch_size": 1,
                "base_model": "NousResearch/Llama-2-7b-hf",
                "base_model_config": "NousResearch/Llama-2-7b-hf",
            }
        )
        cfg = base_cfg | DictDefault(
            {
                "adapter": "lora",
                "lora_r": 128,
                "lora_alpha": 64,
            }
        )
        with self._caplog.at_level(logging.WARNING):
            normalize_config(cfg)
            assert any(
                "soon to be deprecated. please use peft_" in record.message
                for record in self._caplog.records
            )

            assert cfg.peft_r == 128
            assert cfg.peft_alpha == 64
