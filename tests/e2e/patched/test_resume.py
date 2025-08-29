"""
E2E tests for resuming training
"""

import re
import subprocess

from transformers.utils import is_torch_bf16_gpu_available

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, most_recent_subdir, require_torch_2_6_0


class TestResumeLlama:
    """
    Test case for resuming training of llama models
    """

    @require_torch_2_6_0
    def test_resume_lora_packed(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.001,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "save_steps": 3,
                "save_total_limit": 5,
                "max_steps": 15,
                "use_tensorboard": True,
                "save_safetensors": True,
                "save_first_step": False,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)

        resume_cfg = cfg | DictDefault(
            {
                "resume_from_checkpoint": f"{temp_dir}/checkpoint-9/",
            }
        )
        normalize_config(resume_cfg)

        train(cfg=resume_cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        tb_log_path_1 = most_recent_subdir(temp_dir + "/runs")
        cmd = f"tensorboard --inspect  --logdir {tb_log_path_1}"
        res = subprocess.run(
            cmd, shell=True, text=True, capture_output=True, check=True
        )
        pattern = r"first_step\s+(\d+)"
        first_steps = int(re.findall(pattern, res.stdout)[0])
        assert first_steps == 10
