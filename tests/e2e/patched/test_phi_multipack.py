"""
E2E tests for lora llama
"""

import logging
import os
import unittest
from pathlib import Path

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestPhiMultipack(unittest.TestCase):
    """
    Test case for Phi2 models
    """

    @with_temp_dir
    def test_ft_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "microsoft/phi-1_5",
                "model_type": "PhiForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "pad_to_sequence_len": True,
                "load_in_8bit": False,
                "adapter": None,
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
                "dataset_shard_num": 10,
                "dataset_shard_idx": 0,
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "eval_steps": 10,
                "save_steps": 10,
                "bf16": "auto",
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "pytorch_model.bin").exists()

    @with_temp_dir
    def test_qlora_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "microsoft/phi-1_5",
                "model_type": "PhiForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "pad_to_sequence_len": True,
                "load_in_8bit": False,
                "adapter": "qlora",
                "lora_r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
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
                "dataset_shard_num": 10,
                "dataset_shard_idx": 0,
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "eval_steps": 10,
                "save_steps": 10,
                "bf16": "auto",
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()
