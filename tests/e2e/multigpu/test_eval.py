"""
E2E tests for multigpu eval
"""

import logging
import os
from pathlib import Path

import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from ..utils import check_tensorboard

LOG = logging.getLogger("axolotl.tests.e2e.multigpu")
os.environ["WANDB_DISABLED"] = "true"

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


class TestMultiGPUEval:
    """
    Test case for MultiGPU Eval Sample Packing
    """

    def test_eval_sample_packing(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "load_in_8bit": False,
                "load_in_4bit": True,
                "strict": False,
                "sequence_len": 2048,
                "adapter": "qlora",
                "sample_packing": True,
                "eval_sample_packing": True,
                "pad_to_sequence_len": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
                "val_set_size": 0.004,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {
                        "path": "teknium/GPT4-LLM-Cleaned",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "loss_watchdog_threshold": 5.0,
                "loss_watchdog_patience": 3,
                "bf16": "auto",
                "warmup_steps": 1,
                "evals_per_epoch": 2,
                "eval_max_new_tokens": 128,
                "saves_per_epoch": 1,
                "logging_steps": 1,
                "weight_decay": 0.0,
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

        check_tensorboard(temp_dir + "/runs", "eval/loss", 2.5, "Eval Loss is too high")

    def test_eval(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "load_in_8bit": False,
                "load_in_4bit": True,
                "strict": False,
                "sequence_len": 2048,
                "adapter": "qlora",
                "sample_packing": True,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
                "val_set_size": 0.0004,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {
                        "path": "teknium/GPT4-LLM-Cleaned",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "loss_watchdog_threshold": 5.0,
                "loss_watchdog_patience": 3,
                "bf16": "auto",
                "warmup_steps": 1,
                "evals_per_epoch": 2,
                "eval_max_new_tokens": 128,
                "saves_per_epoch": 1,
                "logging_steps": 1,
                "weight_decay": 0.0,
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

        check_tensorboard(temp_dir + "/runs", "eval/loss", 2.9, "Eval Loss is too high")
