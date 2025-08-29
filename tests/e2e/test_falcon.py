"""
E2E tests for falcon
"""

import unittest

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestFalcon(unittest.TestCase):
    """
    Test case for falcon
    """

    @pytest.mark.skip(reason="no tiny models for testing with safetensors")
    @with_temp_dir
    def test_lora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "illuin/tiny-random-FalconForCausalLM",
                "flash_attention": True,
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": [
                    "word_embeddings",
                    "lm_head",
                ],
                "val_set_size": 0.02,
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
                "optimizer": "adamw_torch_fused",
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
    def test_lora_added_vocab(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "illuin/tiny-random-FalconForCausalLM",
                "flash_attention": True,
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": [
                    "word_embeddings",
                    "lm_head",
                ],
                "val_set_size": 0.02,
                "special_tokens": {
                    "bos_token": "<|endoftext|>",
                    "pad_token": "<|endoftext|>",
                },
                "tokens": [
                    "<|im_start|>",
                    "<|im_end|>",
                ],
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
                "optimizer": "adamw_torch_fused",
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
                "sequence_len": 1024,
                "val_set_size": 0.02,
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
                "optimizer": "adamw_torch_fused",
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
