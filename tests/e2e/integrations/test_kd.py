"""
e2e tests for kd trainer support in Axolotl
"""

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async, get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard, require_torch_2_5_1


@pytest.fixture(name="kd_min_cfg")
def min_cfg(temp_dir):
    return {
        "base_model": "Qwen/Qwen3-0.6B",
        "tokenizer_config": "winglian/qwen3-14b-math",
        "plugins": [
            "axolotl.integrations.kd.KDPlugin",
            "axolotl.integrations.liger.LigerPlugin",
        ],
        "liger_rms_norm": True,
        "liger_glu_activation": True,
        "torch_compile": True,
        "chat_template": "qwen3",
        "kd_trainer": True,
        "kd_ce_alpha": 0.1,
        "kd_alpha": 0.9,
        "kd_temperature": 1.0,
        "kd_beta": 0.0,
        "kd_normalize_topk": True,
        "dataloader_prefetch_factor": 8,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "datasets": [
            {
                "path": "winglian/OpenThoughts-114k-math-correct-qwen3-14b-math-prepared-topk128-normalized",
                "type": "chat_template",
                "split": "train",
                "split_thinking": True,
                "eot_tokens": ["<|im_end|>"],
                "data_files": ["train/batch-000000.parquet"],
            },
        ],
        "skip_prepare_dataset": True,
        "val_set_size": 0.0,
        "sequence_len": 2048,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 1,
        "num_epochs": 1,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.00001,
        "bf16": "auto",
        "gradient_checkpointing": True,
        "flash_attention": True,
        "special_tokens": {
            "pad_token": "<|end_of_text|>",
            "eos_token": "<|eot_id|>",
        },
        "max_steps": 5,
        "output_dir": temp_dir,
        "save_safetensors": True,
        "use_tensorboard": True,
        "save_first_step": False,
    }


class TestKnowledgeDistillation:
    """
    Test case for Knowledge Distillation
    """

    # While this will run on torch 2.4.x without torch_compile enabled
    # the VRAM requirement is higher than what is available in CI
    @require_torch_2_5_1
    def test_llama_kd(self, temp_dir, kd_min_cfg):
        cfg = DictDefault(kd_min_cfg)

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
                "1",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        assert (Path(temp_dir) / "model.safetensors").exists()
        check_tensorboard(
            temp_dir + "/runs", "train/loss", 1.4, "Train Loss (%s) is too high"
        )

    @pytest.mark.parametrize(
        "load_in_8bit",
        [True, False],
    )
    def test_llama_lora_kd(self, temp_dir, kd_min_cfg, load_in_8bit):
        cfg = DictDefault(
            {
                "load_in_8bit": load_in_8bit,
                "torch_compile": False,
                "adapter": "lora",
                "peft_use_dora": True,
                "lora_target_linear": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
                "lora_mlp_kernel": False,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
            }
            | kd_min_cfg
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
                "1",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()
        check_tensorboard(
            temp_dir + "/runs", "train/loss", 1.2, "Train Loss (%s) is too high"
        )
