"""
E2E tests for custom optimizers using Llama
"""

import unittest

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import (
    check_model_output_exists,
    check_tensorboard_loss_decreased,
    require_torch_2_5_1,
    require_torch_2_6_0,
    require_torch_2_7_0,
    with_temp_dir,
)


class TestCustomOptimizers(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_optimi_adamw(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
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
                "num_epochs": 1,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "optimi_adamw",
                "max_steps": 5,
                "lr_scheduler": "cosine",
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        assert trainer.optimizer.optimizer.__class__.__name__ == "AdamW"

    @with_temp_dir
    @require_torch_2_5_1
    def test_adopt_adamw(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
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
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adopt_adamw",
                "lr_scheduler": "cosine",
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        assert "ADOPT" in trainer.optimizer.optimizer.__class__.__name__

    @with_temp_dir
    @require_torch_2_5_1
    def test_muon(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
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
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "muon",
                "lr_scheduler": "cosine",
                "weight_decay": 0.01,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        assert "Muon" in trainer.optimizer.optimizer.__class__.__name__

    @with_temp_dir
    @require_torch_2_7_0
    def test_dion(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "dion",
                "dion_lr": 0.01,
                "dion_momentum": 0.95,
                "lr_scheduler": "cosine",
                "weight_decay": 0.01,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        assert "Dion" in trainer.optimizer.optimizer.__class__.__name__

    @with_temp_dir
    def test_fft_schedule_free_adamw(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForCausalLM",
                "sequence_len": 1024,
                "val_set_size": 0.01,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "schedule_free_adamw",
                "lr_scheduler": "constant",
                "max_steps": 10,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    @require_torch_2_6_0
    def test_came_pytorch(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-llama-50m",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-4,
                "optimizer": "came_pytorch",
                "adam_beta3": 0.9999,
                "adam_epsilon2": 1e-16,
                "max_steps": 80,
                "warmup_steps": 5,
                "logging_steps": 1,
                "lr_scheduler": "cosine",
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
            max_initial=4.0,
            max_final=3.0,
        )


@require_torch_2_7_0
@pytest.mark.parametrize(
    "optimizer_name,expected_class,learning_rate",
    [
        ("flash_adamw", "FlashAdamW", 0.00001),
        ("flash_adam", "FlashAdam", 0.00001),
        ("flash_sgd", "FlashSGD", 0.01),
        ("flash_sgdw", "FlashSGDW", 0.01),
        ("flash_lion", "FlashLion", 0.0001),
    ],
)
def test_flash_optimizers(tmp_path, optimizer_name, expected_class, learning_rate):
    pytest.importorskip("flashoptim")
    temp_dir = str(tmp_path)
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "sequence_len": 1024,
            "load_in_8bit": True,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
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
            "num_epochs": 1,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "output_dir": temp_dir,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "max_steps": 5,
            "lr_scheduler": "cosine",
            "save_first_step": False,
        }
    )

    cfg = validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)

    _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
    check_model_output_exists(temp_dir, cfg)
    assert trainer.optimizer.optimizer.__class__.__name__ == expected_class
