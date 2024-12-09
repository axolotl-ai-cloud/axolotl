"""
E2E tests for multigpu lora tinyllama
"""

import logging
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from e2e.utils import check_tensorboard
from huggingface_hub import snapshot_download
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def download_model():
    # download the model
    snapshot_download("HuggingFaceTB/SmolLM2-135M")


class TestMultiGPULlama:
    """
    Test case for Llama models using LoRA
    """

    def test_lora_ddp(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 2048,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 2],
    )
    def test_lora_ddp_packed(self, temp_dir, gradient_accumulation_steps):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
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
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    def test_dpo_lora_ddp(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 2048,
                "sample_packing": False,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                        "type": "chat_template.default",
                        "field_messages": "conversation",
                        "field_chosen": "chosen",
                        "field_rejected": "rejected",
                        "message_field_role": "role",
                        "message_field_content": "content",
                        "roles": {
                            "system": ["system"],
                            "user": ["user"],
                            "assistant": ["assistant"],
                        },
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "warmup_steps": 0,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    def test_dpo_qlora_ddp(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 2048,
                "sample_packing": False,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                        "type": "chat_template.default",
                        "field_messages": "conversation",
                        "field_chosen": "chosen",
                        "field_rejected": "rejected",
                        "message_field_role": "role",
                        "message_field_content": "content",
                        "roles": {
                            "system": ["system"],
                            "user": ["user"],
                            "assistant": ["assistant"],
                        },
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "warmup_steps": 0,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 2],
    )
    def test_fsdp(self, temp_dir, gradient_accumulation_steps):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": gradient_accumulation_steps,
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
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
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
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "fsdp_state_dict_type",
        ["FULL_STATE_DICT", "SHARDED_STATE_DICT"],
    )
    def test_fsdp_packed(self, temp_dir, fsdp_state_dict_type):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
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
                    "fsdp_state_dict_type": fsdp_state_dict_type,
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
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
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    def test_fsdp_qlora_prequant_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/SmolLM2-135M-bnb-nf4-bf16",
                "adapter": "qlora",
                "mean_resizing_embeddings": True,
                "load_in_4bit": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                # "lora_modules_to_save": [
                #     "embed_tokens",
                #     "lm_head",
                # ],
                "sample_packing": True,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:25%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
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
                "use_tensorboard": True,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 2],
    )
    @pytest.mark.parametrize(
        "deepspeed",
        [
            "deepspeed_configs/zero3_bf16.json",
            "deepspeed_configs/zero3_bf16_cpuoffload_all.json",
            # "deepspeed_configs/zero3_bf16_cpuoffload_params.json",
        ],
    )
    @pytest.mark.parametrize(
        "qlora",
        [True, False],
    )
    def test_ds_zero3_packed(
        self, temp_dir, gradient_accumulation_steps, deepspeed, qlora
    ):
        # pylint: disable=duplicate-code
        if qlora:
            adapter = {
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "load_in_4bit": True,
            }
        else:
            adapter = {}
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "deepspeed": str(AXOLOTL_ROOT / deepspeed),
                "use_tensorboard": True,
                **adapter,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 2],
    )
    @pytest.mark.parametrize(
        "qlora",
        [True, False],
    )
    def test_ds_zero2_packed(self, temp_dir, gradient_accumulation_steps, qlora):
        # pylint: disable=duplicate-code
        if qlora:
            adapter = {
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "load_in_4bit": True,
            }
        else:
            adapter = {}
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "deepspeed": str(AXOLOTL_ROOT / "deepspeed_configs/zero2.json"),
                "use_tensorboard": True,
                **adapter,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 2],
    )
    @pytest.mark.parametrize(
        "qlora",
        [True, False],
    )
    def test_ds_zero1_packed(self, temp_dir, gradient_accumulation_steps, qlora):
        # pylint: disable=duplicate-code
        if qlora:
            adapter = {
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "load_in_4bit": True,
            }
        else:
            adapter = {}
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "deepspeed": str(AXOLOTL_ROOT / "deepspeed_configs/zero1.json"),
                "use_tensorboard": True,
                **adapter,
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
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )
