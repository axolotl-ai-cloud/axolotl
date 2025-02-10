"""
E2E tests for llama w/ S2 attn
"""

import logging
import os
import unittest

import pytest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


@pytest.mark.skip(reason="FIXME?")
class TestLlamaShiftedSparseAttention(unittest.TestCase):
    """
    Test case for Llama models using S2 Attn
    """

    @with_temp_dir
    def test_lora_s2_attn(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 16384,
                "sample_packing": False,
                "flash_attention": True,
                "s2_attention": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "special_tokens": {},
                "datasets": [
                    {
                        "path": "Yukang/LongAlpaca-12k",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 10,
                "save_steps": 5,
                "eval_steps": 5,
                "bf16": "auto",
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @with_temp_dir
    def test_fft_s2_attn(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 16384,
                "sample_packing": False,
                "flash_attention": True,
                "s2_attention": True,
                "val_set_size": 0.1,
                "special_tokens": {},
                "datasets": [
                    {
                        "path": "Yukang/LongAlpaca-12k",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 10,
                "save_steps": 5,
                "eval_steps": 5,
                "bf16": "auto",
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
