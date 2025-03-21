"""
Test classes for checking functionality of the cfg normalization
"""

import unittest
from unittest.mock import patch

from axolotl.utils.config import normalize_cfg_datasets, normalize_config
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

    def test_base_model_config_set_when_empty(self):
        cfg = self._get_base_cfg()
        del cfg.base_model_config
        normalize_config(cfg)

        assert cfg.base_model_config == cfg.base_model

    def test_chat_template_chatml(self):
        cfg = DictDefault(
            {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "lorem/ipsum",
                        "type": "chat_template",
                        "chat_template": "gemma",
                    },
                    {
                        "path": "sit/amet",
                        "type": "chat_template",
                    },
                ],
            }
        )

        normalize_cfg_datasets(cfg)

        assert cfg.datasets[0].chat_template == "gemma"
        assert cfg.datasets[1].chat_template == "chatml"

    @patch("axolotl.utils.config.is_torch_bf16_gpu_available")
    def test_bf16_auto_setter_available(self, mock_bf16_avail):
        cfg = self._get_base_cfg()
        cfg.bf16 = "auto"
        mock_bf16_avail.return_value = True

        normalize_config(cfg)

        self.assertTrue(cfg.bf16)
        self.assertFalse(cfg.fp16)

    @patch("axolotl.utils.config.is_torch_bf16_gpu_available")
    def test_bf16_auto_setter_not_available(self, mock_bf16_avail):
        cfg = self._get_base_cfg()
        cfg.bf16 = "auto"
        cfg.fp16 = None
        mock_bf16_avail.return_value = False

        normalize_config(cfg)

        self.assertFalse(cfg.bf16)
        self.assertTrue(cfg.fp16)

    @patch("axolotl.utils.config.is_torch_bf16_gpu_available")
    def test_bf16_disables_fp16(self, mock_bf16_avail):
        cfg = self._get_base_cfg()
        cfg.bf16 = True
        cfg.fp16 = False
        mock_bf16_avail.return_value = True

        normalize_config(cfg)

        self.assertTrue(cfg.bf16)
        self.assertFalse(cfg.fp16)
