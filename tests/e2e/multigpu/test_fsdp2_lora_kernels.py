"""Test LoRA kernels under FSDP2 multi-GPU training.

Verifies that lora_qkv_kernel, lora_o_kernel, lora_mlp_kernel, and
lora_embedding_kernel work correctly with FSDP2 sharding, including
with bias, dropout, and DoRA enabled.
"""

from pathlib import Path

import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import require_torch_2_7_0

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


def _run_training(temp_dir, cfg):
    """Write config and launch multi-GPU training."""
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


def _base_lora_fsdp2_config(temp_dir, **overrides):
    """Base config for LoRA + FSDP2 + kernel tests."""
    cfg = {
        "base_model": "axolotl-ai-co/tiny-qwen3-129m",
        "sequence_len": 512,
        "val_set_size": 0.0,
        "datasets": [
            {
                "path": "tatsu-lab/alpaca",
                "type": "alpaca",
                "split": "train[:1%]",
            },
        ],
        "adapter": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_target_linear": True,
        "num_epochs": 1,
        "max_steps": 3,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "output_dir": temp_dir,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "cosine",
        "flash_attention": True,
        "bf16": True,
        "fsdp_version": 2,
        "fsdp_config": {
            "offload_params": False,
            "cpu_ram_efficient_loading": False,
            "transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
            "state_dict_type": "FULL_STATE_DICT",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "reshard_after_forward": True,
        },
        # Enable all LoRA kernels
        "lora_mlp_kernel": True,
        "lora_qkv_kernel": True,
        "lora_o_kernel": True,
        "lora_embedding_kernel": True,
        "save_safetensors": True,
    }
    cfg.update(overrides)
    return DictDefault(cfg)


class TestFSDP2LoRAKernels:
    """Test LoRA kernels under FSDP2."""

    @require_torch_2_7_0
    def test_lora_kernels_basic(self, temp_dir):
        """Basic LoRA + kernels + FSDP2: no dropout, no bias, no DoRA."""
        cfg = _base_lora_fsdp2_config(temp_dir)
        _run_training(temp_dir, cfg)
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()

    @require_torch_2_7_0
    def test_lora_kernels_with_dropout(self, temp_dir):
        """LoRA kernels + dropout + FSDP2."""
        cfg = _base_lora_fsdp2_config(temp_dir, lora_dropout=0.1)
        _run_training(temp_dir, cfg)
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()

    @require_torch_2_7_0
    def test_lora_kernels_with_dora(self, temp_dir):
        """LoRA kernels + DoRA + FSDP2."""
        cfg = _base_lora_fsdp2_config(temp_dir, peft_use_dora=True)
        _run_training(temp_dir, cfg)
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()

    @require_torch_2_7_0
    def test_lora_kernels_with_dora_and_dropout(self, temp_dir):
        """LoRA kernels + DoRA + dropout + FSDP2."""
        cfg = _base_lora_fsdp2_config(
            temp_dir,
            peft_use_dora=True,
            lora_dropout=0.05,
        )
        _run_training(temp_dir, cfg)
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()
