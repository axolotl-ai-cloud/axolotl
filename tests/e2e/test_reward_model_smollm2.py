"""
E2E tests for reward model lora llama
"""

import logging
import os
import unittest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestRewardModelLoraSmolLM2(unittest.TestCase):
    """
    Test case for Llama reward models using LoRA
    """

    @with_temp_dir
    def test_rm_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForSequenceClassification",
                "num_labels": 1,
                "chat_template": "alpaca",
                "reward_model": True,
                "sequence_len": 2048,
                "pad_to_sequence_len": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "argilla/distilabel-intel-orca-dpo-pairs",
                        "type": "bradley_terry.chat_template",
                        "split": "train[:10%]",
                    },
                ],
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
                "remove_unused_columns": False,
                "max_steps": 10,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "gradient_checkpointing": True,
                "warmup_ratio": 0.1,
                "use_tensorboard": True,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.5, "Train Loss is too high"
        )
        check_model_output_exists(temp_dir, cfg)
