"""
E2E tests for multigpu qwen2
"""

import logging
import os
import unittest
from pathlib import Path

import yaml
from accelerate.test_utils import execute_subprocess_async

from axolotl.utils.dict import DictDefault

from ..utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"


class TestMultiGPUQwen2(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_qlora_fsdp_dpo(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2-1.5B",
                "load_in_4bit": True,
                "rl": "dpo",
                "chat_template": "chatml",
                "sequence_len": 2048,
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "datasets": [
                    {
                        "path": "Intel/orca_dpo_pairs",
                        "split": "train",
                        "type": "chatml.intel",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 100,
                "warmup_steps": 20,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "tf32": True,
                "gradient_checkpointing": True,
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
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )
