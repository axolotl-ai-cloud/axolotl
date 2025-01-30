"""
E2E tests for process reward model w/ lora llama
"""

import logging
import os
import unittest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestProcessRewardSmolLM2(unittest.TestCase):
    """
    Test case for Llama process reward models using LoRA
    """

    @with_temp_dir
    def test_prm(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForTokenClassification",
                "num_labels": 2,
                "process_reward_model": True,
                "sequence_len": 512,
                "val_set_size": 0.0,
                "datasets": [
                    {
                        "path": "trl-lib/math_shepherd",
                        "type": "stepwise_supervised",
                        "step_separator": "\n",
                        "split": "train[:10%]",
                    },
                ],
                "max_steps": 100,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.0005,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "gradient_checkpointing": True,
                "warmup_ratio": 0.1,
                "use_tensorboard": True,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "seed": 42,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.7, "Train Loss (%s) is too high"
        )

        check_model_output_exists(temp_dir, cfg)
