"""Test module for FSDP1 multi-GPU functionality."""

# pylint: disable=duplicate-code

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


class TestFSDP1:
    """Test class for FSDP1 functionality."""

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
                "fsdp_version": "1",
                "fsdp_config": {
                    "fsdp_offload_params": False,
                    "fsdp_cpu_ram_efficient_loading": fsdp_cpu_ram_efficient_loading,
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
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

    @pytest.mark.parametrize(
        "adapter_config",
        [
            {
                "adapter": "lora",
                "load_in_4bit": False,
            },
            {
                "adapter": "qlora",
                "load_in_4bit": True,
            },
        ],
    )
    def test_lora_sft(self, temp_dir, adapter_config):
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
                "adapter": adapter_config["adapter"],
                "load_in_4bit": adapter_config["load_in_4bit"],
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
                "fsdp_version": "1",
                "fsdp_config": {
                    "fsdp_offload_params": False,
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
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
                "fsdp_version": "1",
                "fsdp_config": {
                    "fsdp_offload_params": False,
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
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

    @pytest.mark.parametrize(
        "adapter_config",
        [
            {
                "adapter": "lora",
                "load_in_4bit": False,
            },
            {
                "adapter": "qlora",
                "load_in_4bit": True,
            },
        ],
    )
    def test_dpo_lora(self, temp_dir, adapter_config):
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "load_in_4bit": adapter_config["load_in_4bit"],
                "rl": "dpo",
                "chat_template": "chatml",
                "sequence_len": 2048,
                "adapter": adapter_config["adapter"],
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.01,
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
                "fsdp_version": "1",
                "fsdp_config": {
                    "fsdp_offload_params": False,
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
                },
                "bf16": "auto",
                "tf32": True,
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
