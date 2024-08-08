"""
E2E tests for multigpu lora tinyllama
"""

import logging
import os
import unittest
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async

from axolotl.utils.dict import DictDefault

from ..utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"


class TestMultiGPULlama(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_lora_ddp(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama_v1.1",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 2048,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 100,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

    @with_temp_dir
    def test_lora_ddp_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama_v1.1",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 2048,
                "sample_packing": True,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 50,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

    @with_temp_dir
    def test_fsdp(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama_v1.1",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 100,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp": [
                    "full_shard",
                    "auto_wrap",
                ],
                "fsdp_config": {
                    "fsdp_limit_all_gathers": True,
                    "fsdp_offload_params": False,
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
                    "fsdp_cpu_ram_efficient_loading": False,
                    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                },
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

    @with_temp_dir
    def test_fsdp_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama_v1.1",
                "tokenizer_type": "LlamaTokenizer",
                "sample_packing": True,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 100,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp": [
                    "full_shard",
                    "auto_wrap",
                ],
                "fsdp_config": {
                    "fsdp_limit_all_gathers": True,
                    "fsdp_offload_params": False,
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
                    "fsdp_cpu_ram_efficient_loading": False,
                    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                },
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

    @pytest.mark.skip("disabled due to upstream issue")
    @with_temp_dir
    def test_fsdp_qlora_prequant_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/TinyLlama_v1.1-bnb-nf4-bf16",
                "tokenizer_type": "AutoTokenizer",
                "adapter": "qlora",
                "load_in_4bit": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": [
                    "embed_tokens",
                    "lm_head",
                ],
                "sample_packing": True,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|end_of_text|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:25%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 100,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp": [
                    "full_shard",
                    "auto_wrap",
                ],
                "fsdp_config": {
                    "fsdp_limit_all_gathers": True,
                    "fsdp_offload_params": False,
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                },
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )
