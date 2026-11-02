"""lora_fp32_gradients keeps the LoRA factors of an RL trainer in FP32 under FSDP2.

The RL trainer builder casts the model to torch_dtype after building the trainer under
FSDP; it must leave the FP32 LoRA factors alone, or the adapter trains (and saves) in
BF16 despite lora_fp32_gradients.
"""

from pathlib import Path

import pytest
import torch
import yaml
from accelerate.test_utils import execute_subprocess_async
from safetensors.torch import load_file
from transformers.testing_utils import get_torch_dist_unique_port

from tests.e2e.multigpu.utils import fsdp2_training_config
from tests.e2e.utils import check_tensorboard, requires_flash_attn

pytestmark = [pytest.mark.gpu, requires_flash_attn]


def test_dpo_lora_fp32_gradients_fsdp2(temp_dir):
    cfg = fsdp2_training_config(
        {
            "base_model": "axolotl-ai-co/tiny-qwen2-129m",
            "sequence_len": 1024,
            "rl": "dpo",
            "chat_template": "chatml",
            "datasets": [
                {
                    "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                    "split": "train[:50]",
                    "type": "chat_template.default",
                    "field_messages": "conversation",
                    "field_chosen": "chosen",
                    "field_rejected": "rejected",
                    "message_field_role": "role",
                    "message_field_content": "content",
                    "roles": {
                        "system": ["system"],
                        "user": ["user"],
                        "assistant": ["assistant"],
                    },
                },
            ],
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_linear": True,
            "lora_fp32_gradients": True,
            "max_steps": 2,
            "warmup_steps": 0,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "val_set_size": 0.0,
            "output_dir": temp_dir,
            "dataset_prepared_path": temp_dir + "/last_run_prepared",
            "learning_rate": 1e-5,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "constant",
            "flash_attention": True,
            "bf16": True,
            "use_tensorboard": True,
            "save_first_step": False,
            "fsdp_version": 2,
            "fsdp_config": {
                "offload_params": False,
                "cpu_ram_efficient_loading": False,
                "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                "state_dict_type": "FULL_STATE_DICT",
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "reshard_after_forward": True,
            },
        }
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
            "2",
            "--main-process-port",
            f"{get_torch_dist_unique_port()}",
        ]
    )

    check_tensorboard(
        temp_dir + "/runs", "train/train_loss", 1.0, "Train Loss (%s) is too high"
    )
    adapter = load_file(str(Path(temp_dir) / "adapter_model.safetensors"))
    dtypes = {name: t.dtype for name, t in adapter.items() if "lora_" in name}
    assert dtypes, f"no LoRA tensors in the saved adapter: {sorted(adapter)[:10]}"
    not_fp32 = {name: dtype for name, dtype in dtypes.items() if dtype != torch.float32}
    assert not not_fp32, (
        f"LoRA factors not kept in FP32: {dict(list(not_fp32.items())[:4])}"
    )
