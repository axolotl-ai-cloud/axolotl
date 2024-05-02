"""
E2E tests for lora llama
"""
import json
import logging
import os
import unittest
from pathlib import Path

from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli import do_merge_lora, load_datasets
from axolotl.cli.merge_lora import modify_cfg_for_merge
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestLoraLlama(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "max_steps": 10,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

    @with_temp_dir
    def test_lora_merge(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "max_steps": 10,
                "bf16": "auto",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

        cfg.lora_model_dir = cfg.output_dir
        cfg.load_in_4bit = False
        cfg.load_in_8bit = False
        cfg.flash_attention = False
        cfg.deepspeed = None
        cfg.fsdp = None

        cfg = modify_cfg_for_merge(cfg)
        cfg.merge_lora = True

        cli_args = TrainerCliArgs(merge_lora=True)

        do_merge_lora(cfg=cfg, cli_args=cli_args)
        assert (Path(temp_dir) / "merged/pytorch_model.bin").exists()

        with open(
            Path(temp_dir) / "merged/config.json", "r", encoding="utf-8"
        ) as f_handle:
            config = f_handle.read()
        config = json.loads(config)
        if is_torch_bf16_gpu_available():
            assert config["torch_dtype"] == "bfloat16"
        else:
            assert config["torch_dtype"] == "float16"
