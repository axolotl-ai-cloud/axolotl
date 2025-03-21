"""
e2e tests for unsloth qlora
"""

import logging
import os

import pytest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, check_tensorboard

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


# pylint: disable=duplicate-code
@pytest.mark.skip(
    reason="Unsloth integration will be broken going into latest transformers"
)
class TestUnslothQLoRA:
    """
    Test class for Unsloth QLoRA Llama models
    """

    @pytest.mark.parametrize(
        "sample_packing",
        [True, False],
    )
    def test_unsloth_llama_qlora_fa2(self, temp_dir, sample_packing):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": sample_packing,
                "flash_attention": True,
                "unsloth_lora_mlp": True,
                "unsloth_lora_qkv": True,
                "unsloth_lora_o": True,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "save_steps": 10,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "use_tensorboard": True,
                "bf16": "auto",
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.0, "Train Loss is too high"
        )

    def test_unsloth_llama_qlora_unpacked(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "unsloth_lora_mlp": True,
                "unsloth_lora_qkv": True,
                "unsloth_lora_o": True,
                "sample_packing": False,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "save_steps": 10,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "use_tensorboard": True,
                "bf16": "auto",
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.0, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "sdp_attention",
        [True, False],
    )
    def test_unsloth_llama_qlora_unpacked_no_fa2_fp16(self, temp_dir, sdp_attention):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "unsloth_lora_mlp": True,
                "unsloth_lora_qkv": True,
                "unsloth_lora_o": True,
                "sample_packing": False,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "save_steps": 10,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "sdp_attention": sdp_attention,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "use_tensorboard": True,
                "fp16": True,
            }
        )

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.0, "Train Loss is too high"
        )
