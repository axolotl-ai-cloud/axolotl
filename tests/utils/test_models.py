"""Module for testing models utils file."""


import unittest
from unittest.mock import patch

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model


class ModelsUtilsTest(unittest.TestCase):
    """Testing module for models utils."""

    def test_cfg_throws_error_with_s2_attention_and_sample_packing(self):
        cfg = DictDefault(
            {
                "s2_attention": True,
                "sample_packing": True,
                "base_model": "",
                "model_type": "LlamaForCausalLM",
            }
        )

        # Mock out call to HF hub
        with patch(
            "axolotl.utils.models.load_model_config"
        ) as mocked_load_model_config:
            mocked_load_model_config.return_value = {}
            with pytest.raises(ValueError) as exc:
                # Should error before hitting tokenizer, so we pass in an empty str
                load_model(cfg, tokenizer="")
            assert (
                "shifted-sparse attention does not currently support sample packing"
                in str(exc.value)
            )
