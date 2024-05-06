"""
E2E tests for lora llama
"""

import logging
import os
import unittest
from pathlib import Path

import pytest

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


@pytest.mark.skip(reason="skipping until upstreamed into transformers")
class TestMamba(unittest.TestCase):
    """
    Test case for Mamba models
    """

    @with_temp_dir
    def test_fft(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "state-spaces/mamba-130m",
                "model_type": "MambaLMHeadModel",
                "tokenizer_type": "AutoTokenizer",
                "tokenizer_config": "EleutherAI/gpt-neox-20b",
                "flash_attention": False,
                "sequence_len": 1024,
                "load_in_8bit": False,
                "val_set_size": 0.0,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "gradient_checkpointing": False,
                "num_epochs": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "eval_steps": None,
                "save_safetensors": False,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "pytorch_model.bin").exists()
