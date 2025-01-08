"""
E2E tests for reward model lora llama
"""

import logging
import os
import unittest

from axolotl.common.cli import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestRewardModelLoraLlama(unittest.TestCase):
    """
    Test case for Llama reward models using LoRA
    """

    @with_temp_dir
    def test_rm_fft(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "model_type": "AutoModelForSequenceClassification",
                "tokenizer_type": "LlamaTokenizer",
                "chat_template": "alpaca",
                "reward_model": True,
                "sequence_len": 1024,
                "pad_to_sequence_len": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.0,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "argilla/distilabel-intel-orca-dpo-pairs",
                        "type": "bradley_terry.chat_template",
                    },
                ],
                "remove_unused_columns": False,
                "max_steps": 10,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "gradient_checkpointing": True,
                "warmup_ratio": 0.1,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
