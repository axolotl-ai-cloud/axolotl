"""
E2E tests for llama pretrain
"""

import logging
import os
import unittest
from pathlib import Path

from tbparse import SummaryReader

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import most_recent_subdir, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestEmbeddingsLrScale(unittest.TestCase):
    """
    Test case for embedding_lr*
    """

    @with_temp_dir
    def test_train_w_embedding_lr_scale(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 1024,
                "sample_packing": True,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "max_steps": 5,
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "embedding_lr_scale": 0.5,
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 2.0, "Loss is too high"

    @with_temp_dir
    def test_train_w_embedding_lr(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 1024,
                "sample_packing": True,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "max_steps": 5,
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "embedding_lr": 0.000005,
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 2.0, "Loss is too high"
