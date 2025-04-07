"""
E2E tests for multigpu lora tinyllama
"""

import logging
import os
from pathlib import Path

import pytest
import transformers
import yaml
from accelerate.test_utils import execute_subprocess_async
from huggingface_hub import snapshot_download
from packaging import version
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard, require_torch_2_6_0

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def download_model():
    # download the model
    snapshot_download("HuggingFaceTB/SmolLM2-135M")


def transformers_version_eq(required_version):
    return version.parse(transformers.__version__) == version.parse(required_version)


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
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
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
                        "split": "train[:20%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
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
                "val_set_size": 0.01,
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
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "warmup_steps": 0,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
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

        loss_threshold = 2.3
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            loss_threshold,
            "Train Loss is too high",
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
                "val_set_size": 0.01,
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
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "warmup_steps": 0,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
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

        loss_threshold = 2.3
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            loss_threshold,
            "Train Loss is too high",
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
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
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
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
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
                "sequence_len": 1024,
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
                "gradient_accumulation_steps": 2,
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
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
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @require_torch_2_6_0
    @pytest.mark.parametrize(
        "attention_backend",
        ["flash", "flex"],
    )
    @pytest.mark.parametrize(
        "fsdp_reshard_after_forward",
        [True, False],
    )
    def test_fsdp2_packed(
        self, temp_dir, attention_backend, fsdp_reshard_after_forward
    ):
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
                "gradient_accumulation_steps": 2,
                "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_8bit",
                "lr_scheduler": "cosine",
                "fsdp": [
                    "auto_wrap",
                ],
                "fsdp_config": {
                    "fsdp_version": 2,
                    # "fsdp_forward_prefetch": True,  # not yet implemented in accelerate
                    "fsdp_offload_params": False,
                    "fsdp_cpu_ram_efficient_loading": False,
                    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_reshard_after_forward": fsdp_reshard_after_forward,
                },
                "use_tensorboard": True,
            }
        )
        if attention_backend == "flash":
            cfg.flash_attention = True
        elif attention_backend == "flex":
            cfg.flex_attention = True

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

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.1, "Train Loss is too high"
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
                "sequence_len": 1024,
                "val_set_size": 0.01,
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
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
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
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    # TODO: remove skip once deepspeed regression is fixed
    # see https://github.com/huggingface/transformers/pull/37324
    @pytest.mark.skipif(
        transformers_version_eq("4.51.0"),
        reason="zero3 is not supported with transformers==4.51.0",
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
                "sequence_len": 1024,
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
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
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
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
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
                "sequence_len": 1024,
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
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
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
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
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
                "sequence_len": 1024,
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
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
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
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @pytest.mark.skip(
        reason="fix untrained tokens brittle with lots of edge cases in latest transformers"
    )
    def test_fix_untrained_tokens(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "fix_untrained_tokens": True,
                "sequence_len": 512,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                    "bos_token": "<|custom_im_start|>",
                    "eos_token": "<|custom_im_end|>",
                },
                "datasets": [
                    {
                        "chat_template": "jinja",
                        "chat_template_jinja": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|custom_im_start|>' + message['role'] + '\n' + message['content'] + '<|custom_im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|custom_im_start|>assistant\n' }}{% endif %}",
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                # "gradient_checkpointing": True,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": True,
                "save_safetensors": True,
                # "deepspeed": str(AXOLOTL_ROOT / "deepspeed_configs/zero1.json"),
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

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 4.0, "Train Loss is too high"
        )
