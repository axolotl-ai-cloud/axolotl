"""
Test classes for checking functionality of the cfg normalization
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from axolotl.utils.config import (
    MULTIMODAL_AUTO_MODEL_MAPPING,
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

    @patch("axolotl.utils.config.load_model_config")
    def test_native_diffusion_gemma_uses_text_only_processing(self, load_config):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/diffusion-gemma"
        cfg.diffusion_lm = {"from_causal_lm": False}
        cfg.adapter = "lora"
        cfg.lora_target_modules = [
            r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
            r"\.0\.self_attn\.q_proj$"
        ]
        load_config.return_value = SimpleNamespace(model_type="diffusion_gemma")

        normalize_config(cfg)

        assert cfg.is_multimodal is False
        assert cfg.processor_config is None
        assert cfg.attn_implementation == "flex_attention"

    @patch("axolotl.utils.config.load_model_config")
    def test_native_diffusion_preserves_explicit_attention_fallbacks(self, load_config):
        load_config.return_value = SimpleNamespace(model_type="diffusion_gemma")
        for attn_implementation in ("eager", "sdpa"):
            cfg = self._get_base_cfg()
            cfg.base_model = "local/diffusion-gemma"
            cfg.diffusion_lm = {"from_causal_lm": False}
            cfg.adapter = "lora"
            cfg.attn_implementation = attn_implementation
            cfg.lora_target_modules = [
                r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
                r"\.0\.self_attn\.q_proj$"
            ]

            normalize_config(cfg)

            assert cfg.attn_implementation == attn_implementation

    @patch("axolotl.utils.config.load_model_config")
    def test_native_diffusion_descriptor_rejects_unsupported_attention(
        self, load_config
    ):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/diffusion-gemma"
        cfg.diffusion_lm = {"from_causal_lm": False}
        cfg.adapter = "lora"
        cfg.attn_implementation = "flash_attention_2"
        cfg.lora_target_modules = [
            r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
            r"\.0\.self_attn\.q_proj$"
        ]
        load_config.return_value = SimpleNamespace(model_type="diffusion_gemma")

        with self.assertRaisesRegex(ValueError, "attn_implementation values"):
            normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_native_nemotron_varlen_preserves_config_intent(self, load_config):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/nemotron-diffusion"
        cfg.diffusion_lm = {"from_causal_lm": False}
        cfg.adapter = "lora"
        cfg.trust_remote_code = True
        cfg.attn_implementation = "varlen"
        cfg.lora_target_modules = ["q_proj"]
        cfg.env_capabilities = {"torch_version": "2.14.0"}
        load_config.return_value = SimpleNamespace(
            model_type="nemotron_labs_diffusion", dlm_paradigm="bidirectional"
        )

        normalize_config(cfg)

        assert cfg.attn_implementation == "varlen"

    @patch("axolotl.utils.config.load_model_config")
    def test_varlen_rejects_unsupported_native_layout(self, load_config):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/diffusion-gemma"
        cfg.diffusion_lm = {"from_causal_lm": False}
        cfg.attn_implementation = "varlen"
        load_config.return_value = SimpleNamespace(model_type="diffusion_gemma")

        with self.assertRaisesRegex(ValueError, "full-sequence diffusion"):
            normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_varlen_rejects_dream_legacy_and_ar_paths(self, load_config):
        cases = (
            ("Dream", {"from_causal_lm": False}),
            ("nemotron_labs_diffusion", {"from_causal_lm": True}),
            ("nemotron_labs_diffusion", None),
        )
        for model_type, diffusion_lm in cases:
            with self.subTest(model_type=model_type, diffusion_lm=diffusion_lm):
                cfg = self._get_base_cfg()
                cfg.base_model = f"local/{model_type}"
                cfg.attn_implementation = "varlen"
                cfg.diffusion_lm = diffusion_lm
                load_config.return_value = SimpleNamespace(model_type=model_type)

                with self.assertRaisesRegex(ValueError, "full-sequence diffusion"):
                    normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_varlen_requires_torch_214_public_gqa_api(self, load_config):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/nemotron-diffusion"
        cfg.diffusion_lm = {"from_causal_lm": False}
        cfg.adapter = "lora"
        cfg.trust_remote_code = True
        cfg.attn_implementation = "varlen"
        cfg.lora_target_modules = ["q_proj"]
        cfg.env_capabilities = {"torch_version": "2.13.0"}
        load_config.return_value = SimpleNamespace(
            model_type="nemotron_labs_diffusion", dlm_paradigm="bidirectional"
        )

        with self.assertRaisesRegex(ValueError, "torch >= 2.14"):
            normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_legacy_diffusion_conversion_keeps_attention_unspecified(self, load_config):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/diffusion-gemma"
        cfg.diffusion_lm = {"from_causal_lm": True}
        cfg.plugins = ["axolotl.integrations.diffusion.DiffusionPlugin"]
        cfg.adapter = "lora"
        cfg.lora_target_modules = [
            r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
            r"\.0\.self_attn\.q_proj$"
        ]
        load_config.return_value = SimpleNamespace(model_type="diffusion_gemma")

        normalize_config(cfg)

        assert cfg.attn_implementation is None

    @patch("axolotl.utils.config.load_model_config")
    def test_diffusion_lm_rejects_model_without_diffusion_profile(self, load_config):
        for model_type in ("llama", "qwen2"):
            with self.subTest(model_type=model_type):
                cfg = self._get_base_cfg()
                cfg.diffusion_lm = {"from_causal_lm": False}
                load_config.return_value = SimpleNamespace(model_type=model_type)

                with self.assertRaisesRegex(ValueError, "no native diffusion profile"):
                    normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_from_causal_lm_requires_legacy_diffusion_plugin(self, load_config):
        cfg = self._get_base_cfg()
        cfg.diffusion_lm = {"from_causal_lm": True}
        load_config.return_value = SimpleNamespace(model_type="llama")

        with self.assertRaisesRegex(ValueError, "DiffusionPlugin"):
            normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_from_causal_lm_with_legacy_diffusion_plugin_is_accepted(self, load_config):
        cfg = self._get_base_cfg()
        cfg.diffusion_lm = {"from_causal_lm": True}
        cfg.plugins = ["axolotl.integrations.diffusion:DiffusionPlugin"]
        load_config.return_value = SimpleNamespace(model_type="llama")

        normalize_config(cfg)

        assert cfg.model_config_type == "llama"

    @patch("axolotl.utils.config.load_model_config")
    def test_native_diffusion_gemma_rejects_explicit_multimodal(self, load_config):
        cfg = self._get_base_cfg()
        cfg.base_model = "local/diffusion-gemma"
        cfg.diffusion_lm = {"from_causal_lm": False}
        cfg.is_multimodal = True
        load_config.return_value = SimpleNamespace(model_type="diffusion_gemma")

        with self.assertRaisesRegex(ValueError, "text-only diffusion"):
            normalize_config(cfg)

    @patch("axolotl.utils.config.load_model_config")
    def test_legacy_multimodal_auto_detection_is_unchanged(self, load_config):
        cfg = self._get_base_cfg()
        load_config.return_value = SimpleNamespace(
            model_type=next(iter(MULTIMODAL_AUTO_MODEL_MAPPING))
        )

        normalize_config(cfg)

        assert cfg.is_multimodal is True

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
                    "fsdp_min_num_params": 100000000,
                }
            }
        )

        cfg_without_version = validate_config(cfg_without_version)

        self.assertEqual(cfg_without_version.fsdp_version, 2)
        self.assertEqual(cfg_without_version.fsdp_config.fsdp_version, 2)
        self.assertEqual(
            cfg_without_version.fsdp_config.auto_wrap_policy, "SIZE_BASED_WRAP"
        )
        self.assertEqual(cfg_without_version.fsdp_config.offload_params, True)
        self.assertEqual(cfg_without_version.fsdp_config.min_num_params, 100000000)

        self.assertNotIn("fsdp_auto_wrap_policy", cfg_without_version.fsdp_config)
        self.assertNotIn("fsdp_offload_params", cfg_without_version.fsdp_config)
        self.assertNotIn("fsdp_min_num_params", cfg_without_version.fsdp_config)

    def test_migrate_fsdp_config_no_fsdp_config(self):
        """Test that function doesn't crash when no fsdp_config is present"""
        cfg = self._get_base_cfg()

        cfg = validate_config(cfg)

        self.assertNotIn("fsdp_config", cfg)
        self.assertEqual(cfg.fsdp_version, 2)

    def test_migrate_fsdp_config_empty_fsdp_config(self):
        """Test migration with empty fsdp_config"""
        cfg = self._get_base_cfg() | DictDefault({"fsdp_config": {}})

        cfg = validate_config(cfg)

        self.assertEqual(cfg.fsdp_version, 2)
        self.assertEqual(cfg.fsdp_config, {})

    def test_migrate_fsdp_config_mixed_keys(self):
        """Test migration with a mix of fsdp_ and non-fsdp_ keys"""
        cfg = self._get_base_cfg() | DictDefault(
            {
                "fsdp_config": {
                    "fsdp_version": 2,
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "mixed_precision_policy": "fp16",
                    "activation_checkpointing": True,
                    "fsdp_reshard_after_forward": False,
                }
            }
        )

        cfg = validate_config(cfg)

        self.assertEqual(cfg.fsdp_version, 2)
        self.assertEqual(cfg.fsdp_config.state_dict_type, "FULL_STATE_DICT")
        self.assertEqual(cfg.fsdp_config.reshard_after_forward, False)
        self.assertEqual(cfg.fsdp_config.mixed_precision_policy, "fp16")
        self.assertEqual(cfg.fsdp_config.activation_checkpointing, True)

        # Check original fsdp_ keys are removed
        self.assertNotIn("fsdp_state_dict_type", cfg.fsdp_config)
        self.assertNotIn("fsdp_reshard_after_forward", cfg.fsdp_config)

        self.assertIn("fsdp_version", cfg.fsdp_config)
