"""
e2e tests for GKD / on-policy distillation trainer support in Axolotl.

Smoke tests: the student generates on-policy rollouts (lmbda=1.0), a same-vocab
teacher scores them with the generalized JSD, and the run trains and saves. The
divergence/rollout correctness is covered by the CPU unit tests in
tests/integrations/test_gkd_*.py.
"""

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async, get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard


@pytest.fixture(name="gkd_min_cfg")
def min_cfg(temp_dir):
    return {
        "base_model": "axolotl-ai-co/tiny-llama-50m",
        "plugins": [
            "axolotl.integrations.gkd.GKDPlugin",
        ],
        "gkd_trainer": True,
        "gkd_teacher": "axolotl-ai-co/tiny-llama-50m",
        "gkd_lmbda": 1.0,
        "gkd_beta": 0.5,
        "gkd_temperature": 0.9,
        "gkd_max_new_tokens": 64,
        "chat_template": "llama3",
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
                "split": "train",
            },
        ],
        "train_on_inputs": False,
        "val_set_size": 0.0,
        "sequence_len": 512,
        "sample_packing": False,
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 1,
        "num_epochs": 1,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.00001,
        "bf16": "auto",
        "gradient_checkpointing": True,
        "flash_attention": True,
        "max_steps": 5,
        "output_dir": temp_dir,
        "use_tensorboard": True,
        "save_first_step": False,
    }


class TestGKD:
    """Test case for on-policy distillation (GKD)."""

    def test_gkd_full_finetune(self, temp_dir, gkd_min_cfg):
        cfg = DictDefault(gkd_min_cfg)

        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "1",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        assert (Path(temp_dir) / "model.safetensors").exists()
        check_tensorboard(
            temp_dir + "/runs", "train/loss", 2.0, "Train Loss (%s) is too high"
        )

    def test_gkd_lora(self, temp_dir, gkd_min_cfg):
        cfg = DictDefault(
            {
                "adapter": "lora",
                "lora_target_linear": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
            }
            | gkd_min_cfg
        )

        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "1",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()
