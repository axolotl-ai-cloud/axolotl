"""
E2E tests for packed training
"""

import logging
import os
import unittest

from tbparse import SummaryReader
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import most_recent_subdir, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestPackedLlama(unittest.TestCase):
    """
    Test case for Packed training of llama models
    """

    @with_temp_dir
    def test_loss_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
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
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "use_tensorboard": True,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 2.0, "Loss is too high"
