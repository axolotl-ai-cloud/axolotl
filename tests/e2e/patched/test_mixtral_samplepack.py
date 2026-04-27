"""
E2E tests for mixtral
"""

import unittest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import (
    check_model_output_exists,
    check_tensorboard_loss_decreased,
    with_temp_dir,
)


class TestMixtral(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_qlora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-mixtral-30m",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {},
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 3e-3,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 80,
                "warmup_steps": 5,
                "logging_steps": 1,
                "save_steps": 80,
                "eval_steps": 80,
                "bf16": "auto",
                "save_first_step": False,
                "use_tensorboard": True,
                "seed": 42,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        check_tensorboard_loss_decreased(
            temp_dir + "/runs",
            initial_window=10,
            final_window=10,
            max_initial=6.0,
            max_final=4.7,
        )

    @with_temp_dir
    def test_ft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-mixtral-30m",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {},
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 5e-4,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 80,
                "warmup_steps": 5,
                "logging_steps": 1,
                "save_steps": 80,
                "eval_steps": 80,
                "bf16": "auto",
                "save_first_step": False,
                "use_tensorboard": True,
                "seed": 42,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        check_tensorboard_loss_decreased(
            temp_dir + "/runs",
            initial_window=5,
            final_window=5,
            max_initial=6.0,
            max_final=4.7,
        )
