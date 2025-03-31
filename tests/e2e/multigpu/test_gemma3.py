"""
E2E tests for multigpu lora tinyllama
"""

import logging
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from huggingface_hub import snapshot_download
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def download_model():
    # download the model
    snapshot_download("axolotl-mirrors/gemma-3-4b-pt", repo_type="model")


class TestMultiGPUGemma3:
    """
    Test case for Gemma3 models using LoRA
    """

    def test_lora_ddp_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "axolotl-mirrors/gemma-3-4b-pt",
                "sequence_len": 2048,
                "ddp_find_unused_parameters": True,
                "sample_packing": True,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.0,
                "chat_template": "gemma3",
                "datasets": [
                    {
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
                "micro_batch_size": 4,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {
                    "use_reentrant": False,
                },
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.0001,
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
            temp_dir + "/runs", "train/train_loss", 1.8, "Train Loss is too high"
        )
