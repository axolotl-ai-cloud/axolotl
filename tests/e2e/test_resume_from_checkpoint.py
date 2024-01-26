"""
E2E tests for resuming training from checkpoint
"""

import logging
import os
import unittest

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestDPOLlamaLora(unittest.TestCase):
    """
    Test case for resume training from checkpoint
    """

    @with_temp_dir
    def test_ipo_lora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                "model_type": "LlamaForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "is_llama_derived_model": True,
                "load_in_8bit": False,
                "load_in_4bit": False,
                "strict": False,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                        "style": "chatml",
                    }
                ],
                "val_set_size": 0.05,
                "output_dir": temp_dir,
                "sequence_len": 219,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "gradient_accumulation_steps": 1,
                "micro_batch_size": 1,
                "num_epochs": 4,
                "max_steps": 20,
                "optimizer": "paged_adamw_32bit",
                "lr_scheduler": "cosine",
                "learning_rate": 0.0002,
                "bf16": False,
                "fp16": False,
                "tf32": False,
                "gradient_checkpointing": True,
                "logging_steps": 1,
                "flash_attention": False,
                "warmup_steps": 10,
                "save_steps": 10,
                "weight_decay": 0.0,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        # TODO check so that there is a path to checkpoints

        cfg["resume_from_checkpoint"] = True

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)

        # TODO check so it started from the checkpoint
