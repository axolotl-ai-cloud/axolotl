"""
E2E tests for lora llama
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


class TestPhiMultipack(unittest.TestCase):
    """
    Test case for Phi2 models
    """

    @with_temp_dir
    def test_ft_packed(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-phi-64m",
                "model_type": "PhiForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "pad_to_sequence_len": True,
                "load_in_8bit": False,
                "adapter": None,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "dataset_shard_num": 10,
                "dataset_shard_idx": 0,
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 2e-4,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 50,
                "logging_steps": 1,
                "eval_steps": 50,
                "save_steps": 50,
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

    @with_temp_dir
    def test_qlora_packed(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-phi-64m",
                "model_type": "PhiForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "pad_to_sequence_len": True,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.02,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "dataset_shard_num": 10,
                "dataset_shard_idx": 0,
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 2e-4,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 50,
                "logging_steps": 1,
                "eval_steps": 50,
                "save_steps": 50,
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
