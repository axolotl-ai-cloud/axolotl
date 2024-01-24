"""
E2E tests for mixtral
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
                "tokenizer_config": "mistralai/Mixtral-8x7B-v0.1",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_linear": True,
                "val_set_size": 0.1,
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

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

    @with_temp_dir
    def test_ft(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "mistralai/Mixtral-8x7B-v0.1",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.1,
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

        model, _ = train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (
            "MixtralFlashAttention2"
            in model.model.layers[0].self_attn.__class__.__name__
        )
        assert (Path(temp_dir) / "pytorch_model.bin").exists()
