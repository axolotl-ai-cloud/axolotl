"""Test module for FSDP2 multi-GPU functionality."""

# pylint: disable=duplicate-code

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import require_torch_2_7_0

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


class TestFSDP2:
    """Test class for FSDP2 functionality."""

    @require_torch_2_7_0
    @pytest.mark.parametrize(
        "fsdp_cpu_ram_efficient_loading",
        [True, False],
    )
    def test_fft_sft(self, temp_dir, fsdp_cpu_ram_efficient_loading):
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": fsdp_cpu_ram_efficient_loading,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "bf16": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

    @require_torch_2_7_0
    def test_lora_sft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "bf16": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

    @require_torch_2_7_0
    def test_qlora_sft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "bf16": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

    @require_torch_2_7_0
    def test_dpo_fft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "Intel/orca_dpo_pairs",
                        "split": "train",
                        "type": "chatml.intel",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

    @require_torch_2_7_0
    def test_dpo_lora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "sequence_len": 2048,
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "Intel/orca_dpo_pairs",
                        "split": "train",
                        "type": "chatml.intel",
                    },
                ],
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )
