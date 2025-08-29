"""Test module for FSDP1 multi-GPU functionality."""

import os
from pathlib import Path

import pytest
import torch
import yaml
from accelerate.test_utils import execute_subprocess_async
from tbparse import SummaryReader
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import most_recent_subdir

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


def verify_training_success(temp_dir):
    """Verify that training completed successfully by checking artifacts and loss."""
    output_path = Path(temp_dir)

    model_files = list(output_path.glob("*.bin")) + list(
        output_path.glob("*.safetensors")
    )
    assert len(model_files) > 0, "No model files found - training may have failed"

    checkpoint_files = list(output_path.glob("checkpoint-*"))
    assert len(checkpoint_files) > 0, (
        "No checkpoint files found - training may have failed"
    )

    tb_log_path = most_recent_subdir(temp_dir + "/runs")
    if tb_log_path:
        event_files = sorted(os.listdir(tb_log_path))
        if event_files:
            event_file = os.path.join(tb_log_path, event_files[0])
            reader = SummaryReader(event_file)
            df = reader.scalars
            train_loss_df = df[df.tag == "train/train_loss"]
            if len(train_loss_df) > 0:
                final_loss = train_loss_df.value.values[-1]
                assert not torch.isnan(torch.tensor(final_loss)), (
                    f"Training loss is NaN: {final_loss}"
                )


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

        verify_training_success(temp_dir)

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

        verify_training_success(temp_dir)

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
                "use_tensorboard": True,
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

        verify_training_success(temp_dir)

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
                "use_tensorboard": True,
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

        verify_training_success(temp_dir)
