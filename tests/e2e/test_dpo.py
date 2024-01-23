"""
E2E tests for lora llama
"""

import logging
import os
import unittest
from pathlib import Path

from axolotl.cli import load_rl_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import with_temp_dir

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
                        "path": "Intel/orca_dpo_pairs",
                        "type": "chatml.intel",
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
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "checkpoint-20/adapter_model.safetensors").exists()

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
                        "path": "Intel/orca_dpo_pairs",
                        "type": "chatml.intel",
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
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "checkpoint-20/adapter_model.safetensors").exists()

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
                        "path": "Intel/orca_dpo_pairs",
                        "type": "chatml.intel",
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
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "checkpoint-20/adapter_model.safetensors").exists()
