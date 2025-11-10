"""
E2E tests for mixtral
"""

import unittest

import torch
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestMixtral(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_qlora_w_fa2(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": True,
                "sequence_len": 1024,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "lora_target_modules": [
                    "o_proj",
                    "w3",
                    "k_proj",
                    "v_proj",
                    "w1",
                    "q_proj",
                    "w2",
                ],
                "val_set_size": 0.02,
                "special_tokens": {},
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
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        model, _, _ = train(cfg=cfg, dataset_meta=dataset_meta)
        assert (
            model.base_model.model.model.layers[0].block_sparse_moe.gate.weight.dtype
            == torch.float32
        )
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_qlora_wo_fa2(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": False,
                "sequence_len": 1024,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "lora_target_modules": [
                    "o_proj",
                    "w3",
                    "k_proj",
                    "v_proj",
                    "w1",
                    "q_proj",
                    "w2",
                ],
                "val_set_size": 0.02,
                "special_tokens": {},
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
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        model, _, _ = train(cfg=cfg, dataset_meta=dataset_meta)
        assert (
            model.base_model.model.model.layers[0].block_sparse_moe.gate.weight.dtype
            == torch.float32
        )
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_16bit_lora_w_fa2(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": True,
                "sequence_len": 1024,
                "adapter": "lora",
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "lora_target_modules": [
                    "o_proj",
                    "w3",
                    "k_proj",
                    "v_proj",
                    "w1",
                    "q_proj",
                    "w2",
                ],
                "val_set_size": 0.02,
                "special_tokens": {},
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
                "save_first_step": False,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        model, _, _ = train(cfg=cfg, dataset_meta=dataset_meta)
        assert (
            model.base_model.model.model.layers[0].block_sparse_moe.gate.weight.dtype
            == torch.float32
        )
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_16bit_lora_wo_fa2(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": False,
                "sequence_len": 1024,
                "adapter": "lora",
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "lora_target_modules": [
                    "o_proj",
                    "w3",
                    "k_proj",
                    "v_proj",
                    "w1",
                    "q_proj",
                    "w2",
                ],
                "val_set_size": 0.02,
                "special_tokens": {},
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
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True
        dataset_meta = load_datasets(cfg=cfg)

        model, _, _ = train(cfg=cfg, dataset_meta=dataset_meta)
        assert (
            model.base_model.model.model.layers[0].block_sparse_moe.gate.weight.dtype
            == torch.float32
        )
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_ft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": True,
                "sequence_len": 1024,
                "val_set_size": 0.02,
                "special_tokens": {},
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
                "save_first_step": False,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
