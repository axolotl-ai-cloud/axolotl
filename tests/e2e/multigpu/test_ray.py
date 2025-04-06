"""
E2E tests for multigpu post-training use Ray Train
"""

import logging
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard, require_torch_lt_2_6_0

LOG = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


class TestMultiGPURay:
    """
    Test cases for AnyScale Ray post training
    """

    @require_torch_lt_2_6_0
    def test_lora_ddp(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
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
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
                "use_ray": True,
                "ray_num_workers": 2,
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
                "--use-ray",
                "--ray-num-workers",
                "2",
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )

    @require_torch_lt_2_6_0
    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 2],
    )
    def test_ds_zero2_packed(self, temp_dir, gradient_accumulation_steps):
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
                "micro_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "deepspeed": str(AXOLOTL_ROOT / "deepspeed_configs/zero2.json"),
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
                "--use-ray",
                "--ray-num-workers",
                "2",
            ]
        )

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 2.3, "Train Loss is too high"
        )
