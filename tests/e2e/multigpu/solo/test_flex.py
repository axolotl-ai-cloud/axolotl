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
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard, require_torch_2_6_0

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def download_model():
    # download the model
    snapshot_download("HuggingFaceTB/SmolLM2-135M")


class TestPackedFlex:
    """
    Test case for Packed training of llama models
    """

    @require_torch_2_6_0
    def test_loss_llama(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "flex_attention": True,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "vicgalle/alpaca-gpt4",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "use_tensorboard": True,
                "save_strategy": "no",
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

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
            temp_dir + "/runs", "train/train_loss", 2.0, "Train Loss is too high"
        )
