"""
E2E tests for deepseekv3
"""

from pathlib import Path

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline


class TestDeepseekV3:
    """
    Test case for DeepseekV3 models
    """

    @enable_hf_offline
    @pytest.mark.parametrize(
        "sample_packing",
        [True, False],
    )
    def test_lora_deepseekv3(self, temp_dir, sample_packing):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/DeepSeek-V3-11M",
                "trust_remote_code": True,
                "sample_packing": sample_packing,
                "flash_attention": True,
                "sequence_len": 2048,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0,
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "message_property_mappings": {
                            "role": "from",
                            "content": "value",
                        },
                        "drop_system_message": True,
                        "split": "train[:1%]",
                    },
                ],
                "special_tokens": {
                    "bos_token": "<｜begin▁of▁sentence｜>",
                    "eos_token": "<｜end▁of▁sentence｜>",
                },
                "chat_template": "deepseek_v3",
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_safetensors": True,
                "bf16": True,
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()

    @enable_hf_offline
    @pytest.mark.parametrize(
        "sample_packing",
        [True, False],
    )
    def test_fft_deepseekv3(self, temp_dir, sample_packing):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/DeepSeek-V3-11M",
                "trust_remote_code": True,
                "sample_packing": sample_packing,
                "flash_attention": True,
                "sequence_len": 2048,
                "val_set_size": 0,
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                        "split": "train[:1%]",
                    },
                ],
                "chat_template": "deepseek_v3",
                "special_tokens": {
                    "bos_token": "<｜begin▁of▁sentence｜>",
                    "eos_token": "<｜end▁of▁sentence｜>",
                },
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_safetensors": True,
                "bf16": True,
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()
