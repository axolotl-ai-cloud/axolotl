"""
E2E tests for falcon
"""

import unittest

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, with_temp_dir


class TestFalconPatched(unittest.TestCase):
    """
    Test case for Falcon models
    """

    @pytest.mark.skip(reason="no tiny models for testing with safetensors")
    @with_temp_dir
    def test_qlora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "illuin/tiny-random-FalconForCausalLM",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "lora_modules_to_save": ["word_embeddings", "lm_head"],
                "val_set_size": 0.05,
                "special_tokens": {
                    "bos_token": "<|endoftext|>",
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "eval_steps": 10,
                "bf16": "auto",
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @pytest.mark.skip(reason="no tiny models for testing with safetensors")
    @with_temp_dir
    def test_ft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "illuin/tiny-random-FalconForCausalLM",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "bos_token": "<|endoftext|>",
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "eval_steps": 10,
                "bf16": "auto",
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
