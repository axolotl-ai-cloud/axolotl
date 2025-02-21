"""
E2E tests for qwen
"""

import logging
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.tests.qwen")
os.environ["WANDB_DISABLED"] = "true"


class TestE2eQwen:
    """
    Test cases for qwen models
    """

    @pytest.mark.parametrize("base_model", ["Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B"])
    def test_dpo(self, base_model, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": base_model,
                "rl": "dpo",
                "chat_template": "qwen_25",
                "sequence_len": 2048,
                "val_set_size": 0.0,
                "datasets": [
                    {
                        "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                        "split": "train",
                        "type": "chat_template.default",
                        "field_messages": "conversation",
                        "field_chosen": "chosen",
                        "field_rejected": "rejected",
                        "message_property_mappings": {
                            "role": "role",
                            "content": "content",
                        },
                        "roles": {
                            "system": ["system"],
                            "user": ["user"],
                            "assistant": ["assistant"],
                        },
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "warmup_steps": 20,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "tf32": True,
                "gradient_checkpointing": True,
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
