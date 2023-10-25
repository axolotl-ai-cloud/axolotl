"""
Test classes for checking functionality of the cfg normalization
"""
import unittest

from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault


class NormalizeConfigTestCase(unittest.TestCase):
    """
    test class for normalize_config checks
    """

    def _get_base_cfg(self):
        return DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "base_model_config": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        )

    def test_lr_as_float(self):
        cfg = (
            self._get_base_cfg()
            | DictDefault(  # pylint: disable=unsupported-binary-operation
                {
                    "learning_rate": "5e-5",
                }
            )
        )

        normalize_config(cfg)

        assert cfg.learning_rate == 0.00005

    def test_base_model_config_set_when_empty(self):
        cfg = self._get_base_cfg()
        del cfg.base_model_config
        normalize_config(cfg)

        assert cfg.base_model_config == cfg.base_model
