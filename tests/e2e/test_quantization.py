"""
E2E tests for training with quantized model
"""

import logging
import os
import unittest

from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_tensorboard, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestHQQ(unittest.TestCase):
    """
    Test cases for training of HQQ-quantized llama models"""

    @with_temp_dir
    def test_hqq_qlora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "use_hqq": True,
                "hqq_config": [
                    {
                        "nbits": 8,
                        "group_size": 32,
                    }
                ],
                "adapter": "lora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "vicgalle/alpaca-gpt4",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "use_tensorboard": True,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.0, "Train Loss is too high"
        )
