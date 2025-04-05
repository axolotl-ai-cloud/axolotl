"""
E2E tests for multigpu qwen2
"""

import logging
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"


class TestMultiGPUQwen2:
    """
    Test case for Llama models using LoRA
    """

    @pytest.mark.parametrize("base_model", ["Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B"])
    def test_qlora_fsdp_dpo(self, base_model, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": base_model,
                "load_in_4bit": True,
                "rl": "dpo",
                "chat_template": "chatml",
                "sequence_len": 2048,
                "adapter": "qlora",
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
                "warmup_steps": 20,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "tf32": True,
                # "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {
                    "use_reentrant": False,
                },
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
                    "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_sharding_strategy": "FULL_SHARD",
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
