"""
E2E tests for multipack fft llama using 4d attention masks
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


class Test4dMultipackLlama(unittest.TestCase):
    """
    Test case for Llama models using 4d attention with multipack
    """

    @with_temp_dir
    def test_sdp_lora_packing(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "flash_attention": False,
                "sdp_attention": True,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "sequence_len": 1024,
                "val_set_size": 0.02,
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
                "fp16": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_torch_lora_packing(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "flash_attention": False,
                "sdp_attention": False,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.02,
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
                "fp16": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
