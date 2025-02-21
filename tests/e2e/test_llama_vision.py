"""
E2E tests for lora llama
"""

import logging
import os
import unittest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestLlamaVision(unittest.TestCase):
    """
    Test case for Llama Vision models
    """

    @with_temp_dir
    def test_lora_llama_vision_text_only_dataset(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/Llama-3.2-39M-Vision",
                "processor_type": "AutoProcessor",
                "skip_prepare_dataset": True,
                "remove_unused_columns": False,
                "sample_packing": False,
                "sequence_len": 1024,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": r"language_model.model.layers.[\d]+.(mlp|cross_attn|self_attn).(up|down|gate|q|k|v|o)_proj",
                "val_set_size": 0,
                "chat_template": "llama3_2_vision",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_safetensors": True,
                "bf16": True,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_lora_llama_vision_multimodal_dataset(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/Llama-3.2-39M-Vision",
                "processor_type": "AutoProcessor",
                "skip_prepare_dataset": True,
                "remove_unused_columns": False,
                "sample_packing": False,
                "sequence_len": 1024,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": r"language_model.model.layers.[\d]+.(mlp|cross_attn|self_attn).(up|down|gate|q|k|v|o)_proj",
                "val_set_size": 0,
                "chat_template": "llama3_2_vision",
                "datasets": [
                    {
                        "path": "axolotl-ai-co/llava-instruct-mix-vsft-small",
                        "type": "chat_template",
                        "split": "train",
                        "field_messages": "messages",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_safetensors": True,
                "bf16": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
