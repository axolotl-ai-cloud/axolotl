"""
E2E tests for resuming training
"""

import logging
import os
import re
import subprocess
import unittest
from pathlib import Path

from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import most_recent_subdir, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestResumeLlama(unittest.TestCase):
    """
    Test case for resuming training of llama models
    """

    @with_temp_dir
    def test_resume_qlora_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "special_tokens": {},
                "datasets": [
                    {
                        "path": "vicgalle/alpaca-gpt4",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "save_steps": 10,
                "save_total_limit": 5,
                "max_steps": 40,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)

        resume_cfg = cfg | DictDefault(
            {
                "resume_from_checkpoint": f"{temp_dir}/checkpoint-30/",
            }
        )
        normalize_config(resume_cfg)
        cli_args = TrainerCliArgs()

        train(cfg=resume_cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

        tb_log_path_1 = most_recent_subdir(temp_dir + "/runs")
        cmd = f"tensorboard --inspect  --logdir {tb_log_path_1}"
        res = subprocess.run(
            cmd, shell=True, text=True, capture_output=True, check=True
        )
        pattern = r"first_step\s+(\d+)"
        first_steps = int(re.findall(pattern, res.stdout)[0])
        assert first_steps == 31
