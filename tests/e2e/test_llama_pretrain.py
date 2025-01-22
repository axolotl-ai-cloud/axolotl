"""
E2E tests for llama pretrain
"""

import logging
import os

import pytest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestPretrainLlama:
    """
    Test case for Llama models w pretraining
    """

    @pytest.mark.parametrize(
        "sample_packing",
        [True, False],
    )
    @pytest.mark.parametrize(
        "pretrain_multipack_attn",
        [True, False],
    )
    def test_pretrain(self, temp_dir, sample_packing, pretrain_multipack_attn):
        if not sample_packing and pretrain_multipack_attn:
            return

        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 2048,
                "sample_packing": sample_packing,
                "pretrain_multipack_attn": pretrain_multipack_attn,
                "dataset_processes": 1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "pretraining_dataset": [
                    {
                        "path": "allenai/c4",
                        "name": "en",
                        "type": "pretrain",
                    }
                ],
                "max_steps": 5,
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 4.8, "Train Loss is too high"
        )
