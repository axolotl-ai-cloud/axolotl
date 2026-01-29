"""
Test classes for checking functionality of the cfg normalization
"""

import unittest
from unittest.mock import patch

from axolotl.utils.config import (
    normalize_cfg_datasets,
    normalize_config,
    validate_config,
)
from axolotl.utils.dict import DictDefault


class NormalizeConfigTestCase(unittest.TestCase):
    """
    test class for normalize_config checks
    """

    def _get_base_cfg(self):
        return DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "base_model_config": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "learning_rate": 0.0001,
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

    def test_migrate_fsdp_config(self):
        """Test basic FSDP config migration with and without fsdp_version"""
        cfg_with_version = self._get_base_cfg() | DictDefault(
            {
                "fsdp_config": {
                    "fsdp_version": 2,
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_offload_params": False,
                    "fsdp_cpu_ram_efficient_loading": True,
                }
            }
        )

        cfg_with_version = validate_config(cfg_with_version)

        self.assertEqual(cfg_with_version.fsdp_version, 2)
        self.assertEqual(
            cfg_with_version.fsdp_config.auto_wrap_policy, "TRANSFORMER_BASED_WRAP"
        )
        self.assertEqual(cfg_with_version.fsdp_config.offload_params, False)
        self.assertEqual(cfg_with_version.fsdp_config.cpu_ram_efficient_loading, True)

        self.assertNotIn("fsdp_auto_wrap_policy", cfg_with_version.fsdp_config)
        self.assertNotIn("fsdp_offload_params", cfg_with_version.fsdp_config)
        self.assertNotIn("fsdp_cpu_ram_efficient_loading", cfg_with_version.fsdp_config)
        self.assertIn("fsdp_version", cfg_with_version.fsdp_config)

        cfg_without_version = self._get_base_cfg() | DictDefault(
            {
                "fsdp_config": {
                    "fsdp_auto_wrap_policy": "SIZE_BASED_WRAP",
                    "fsdp_offload_params": True,
                }
            }
        )

        cfg_without_version = validate_config(cfg_without_version)

        self.assertNotIn("fsdp_version", cfg_without_version)
        self.assertEqual(
            cfg_without_version.fsdp_config.auto_wrap_policy, "SIZE_BASED_WRAP"
        )
        self.assertEqual(cfg_without_version.fsdp_config.offload_params, True)

        self.assertNotIn("fsdp_auto_wrap_policy", cfg_without_version.fsdp_config)
        self.assertNotIn("fsdp_offload_params", cfg_without_version.fsdp_config)

    def test_migrate_fsdp_config_no_fsdp_config(self):
        """Test that function doesn't crash when no fsdp_config is present"""
        cfg = self._get_base_cfg()

        cfg = validate_config(cfg)

        self.assertNotIn("fsdp_config", cfg)
        self.assertNotIn("fsdp_version", cfg)

    def test_migrate_fsdp_config_empty_fsdp_config(self):
        """Test migration with empty fsdp_config"""
        cfg = self._get_base_cfg() | DictDefault({"fsdp_config": {}})

        cfg = validate_config(cfg)

        self.assertNotIn("fsdp_version", cfg)
        self.assertEqual(cfg.fsdp_config, {})

    def test_migrate_fsdp_config_mixed_keys(self):
        """Test migration with a mix of fsdp_ and non-fsdp_ keys"""
        cfg = self._get_base_cfg() | DictDefault(
            {
                "fsdp_config": {
                    "fsdp_version": 1,
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "mixed_precision_policy": "fp16",
                    "activation_checkpointing": True,
                    "fsdp_reshard_after_forward": False,
                }
            }
        )

        cfg = validate_config(cfg)

        self.assertEqual(cfg.fsdp_version, 1)
        self.assertEqual(cfg.fsdp_config.state_dict_type, "FULL_STATE_DICT")
        self.assertEqual(cfg.fsdp_config.reshard_after_forward, False)
        self.assertEqual(cfg.fsdp_config.mixed_precision_policy, "fp16")
        self.assertEqual(cfg.fsdp_config.activation_checkpointing, True)

        # Check original fsdp_ keys are removed
        self.assertNotIn("fsdp_state_dict_type", cfg.fsdp_config)
        self.assertNotIn("fsdp_reshard_after_forward", cfg.fsdp_config)

        self.assertIn("fsdp_version", cfg.fsdp_config)
