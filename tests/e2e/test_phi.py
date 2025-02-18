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


class TestPhi(unittest.TestCase):
    """
    Test case for Phi2 models
    """

    @with_temp_dir
    def test_phi_ft(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "microsoft/phi-1_5",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 2048,
                "sample_packing": False,
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
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "max_steps": 10,
                "save_steps": 10,
                "eval_steps": 10,
                "bf16": "auto",
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_phi_qlora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "microsoft/phi-1_5",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 2048,
                "sample_packing": False,
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
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "max_steps": 10,
                "save_steps": 10,
                "eval_steps": 10,
                "bf16": "auto",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
