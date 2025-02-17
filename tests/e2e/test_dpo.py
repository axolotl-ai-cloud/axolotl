"""
E2E tests for lora llama
"""

import logging
import os
import unittest
from pathlib import Path

import pytest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_preference_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestDPOLlamaLora(unittest.TestCase):
    """
    Test case for DPO Llama models using LoRA
    """

    @with_temp_dir
    def test_dpo_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "dpo",
                "datasets": [
                    {
                        "path": "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized",
                        "type": "chatml.ultra",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)

    @with_temp_dir
    def test_dpo_nll_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "dpo",
                "rpo_alpha": 0.5,
                "datasets": [
                    {
                        "path": "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized",
                        "type": "chatml.ultra",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)

    @with_temp_dir
    def test_dpo_use_weighting(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "dpo",
                "dpo_use_weighting": True,
                "datasets": [
                    {
                        "path": "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized",
                        "type": "chatml.ultra",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)

    @pytest.mark.skip("kto_pair no longer supported in trl")
    @with_temp_dir
    def test_kto_pair_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "kto_pair",
                "datasets": [
                    {
                        "path": "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized",
                        "type": "chatml.ultra",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)

    @with_temp_dir
    def test_ipo_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "ipo",
                "datasets": [
                    {
                        "path": "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized",
                        "type": "chatml.ultra",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)

    @with_temp_dir
    def test_orpo_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "orpo",
                "orpo_alpha": 0.1,
                "remove_unused_columns": False,
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "argilla/distilabel-capybara-dpo-7k-binarized",
                        "type": "chat_template.argilla",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)

    @pytest.mark.skip(reason="Fix the implementation")
    @with_temp_dir
    def test_kto_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "special_tokens": {},
                "rl": "kto",
                "rl_beta": 0.5,
                "kto_desirable_weight": 1.0,
                "kto_undesirable_weight": 1.0,
                "remove_unused_columns": False,
                "datasets": [
                    # {
                    #     "path": "argilla/kto-mix-15k",
                    #     "type": "chatml.argilla_chat",
                    #     "split": "train",
                    # },
                    {
                        "path": "argilla/ultrafeedback-binarized-preferences-cleaned-kto",
                        "type": "chatml.ultra",
                        "split": "train",
                    },
                    # {
                    #     "path": "argilla/kto-mix-15k",
                    #     "type": "llama3.argilla_chat",
                    #     "split": "train",
                    # },
                    {
                        "path": "argilla/ultrafeedback-binarized-preferences-cleaned-kto",
                        "type": "llama3.ultra",
                        "split": "train",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "warmup_steps": 5,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-20", cfg)
