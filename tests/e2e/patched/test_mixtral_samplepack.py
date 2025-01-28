"""
E2E tests for mixtral
"""

import logging
import os
import unittest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestMixtral(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_qlora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "val_set_size": 0.05,
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
                "bf16": "auto",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_ft(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
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
                "bf16": "auto",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
