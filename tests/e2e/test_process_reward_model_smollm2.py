"""
E2E tests for process reward model w/ lora llama
"""

import unittest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard, with_temp_dir


class TestProcessRewardSmolLM2(unittest.TestCase):
    """
    Test case for Llama process reward models using LoRA
    """

    @with_temp_dir
    def test_prm(self, temp_dir):
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
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.7, "Train Loss (%s) is too high"
        )

        check_model_output_exists(temp_dir, cfg)
