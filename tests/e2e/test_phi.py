"""
E2E tests for lora llama
"""

import logging
import os
import tempfile
import unittest

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestPhi(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    def test_ft(self):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "microsoft/phi-1_5",
                "base_model_config": "microsoft/phi-1_5",
                "trust_remote_code": True,
                "model_type": "MixFormerSequentialForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 512,
                "sample_packing": False,
                "load_in_8bit": False,
                "adapter": None,
                "val_set_size": 0.1,
                "special_tokens": {
                    "unk_token": "<|endoftext|>",
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
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
                "output_dir": tempfile.mkdtemp(),
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "bf16": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)

    def test_ft_packed(self):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "microsoft/phi-1_5",
                "base_model_config": "microsoft/phi-1_5",
                "trust_remote_code": True,
                "model_type": "MixFormerSequentialForCausalLM",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 512,
                "sample_packing": True,
                "load_in_8bit": False,
                "adapter": None,
                "val_set_size": 0.1,
                "special_tokens": {
                    "unk_token": "<|endoftext|>",
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
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
                "output_dir": tempfile.mkdtemp(),
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "bf16": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
