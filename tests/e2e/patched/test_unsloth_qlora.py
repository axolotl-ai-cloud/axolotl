"""
e2e tests for unsloth qlora
"""
import logging
import os
from pathlib import Path

import pytest
from e2e.utils import most_recent_subdir
from tbparse import SummaryReader

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


# pylint: disable=duplicate-code
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
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.2,
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

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 2.0, "Loss is too high"

    def test_unsloth_llama_qlora_unpacked(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": False,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.2,
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

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 2.0, "Loss is too high"

    @pytest.mark.parametrize(
        "sdp_attention",
        [True, False],
    )
    def test_unsloth_llama_qlora_unpacked_no_fa2_fp16(self, temp_dir, sdp_attention):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": False,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.2,
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

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 2.0, "Loss is too high"
