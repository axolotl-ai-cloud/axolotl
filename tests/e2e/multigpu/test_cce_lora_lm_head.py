"""LoRA on lm_head under cut_cross_entropy must train when the head is sharded (FSDP2, ZeRO-3)."""

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from safetensors.torch import load_file
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


def _base_cfg(temp_dir):
    return {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
        "cut_cross_entropy": True,
        "adapter": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj", "v_proj", "lm_head"],
        "sequence_len": 512,
        "val_set_size": 0.0,
        "special_tokens": {"pad_token": "<|endoftext|>"},
        "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
        "num_epochs": 1,
        "max_steps": 2,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-4,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "output_dir": temp_dir,
        "dataset_prepared_path": temp_dir + "/last_run_prepared",
        "save_first_step": False,
        "seed": 42,
    }


def _run(temp_dir, cfg):
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(DictDefault(cfg).to_dict(), Dumper=yaml.Dumper))
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
    # lora_B starts at zero, so a head adapter CCE never saw is saved as all zeros.
    adapter = load_file(str(Path(temp_dir) / "adapter_model.safetensors"))
    lm_head_b = [v for k, v in adapter.items() if "lm_head" in k and "lora_B" in k]
    assert lm_head_b, list(adapter)
    assert all(t.float().abs().sum() > 0 for t in lm_head_b)


class TestCCELoraLmHeadMultiGPU:
    def test_fsdp2(self, temp_dir):
        _run(
            temp_dir,
            _base_cfg(temp_dir)
            | {
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
            },
        )

    @pytest.mark.parametrize(
        "deepspeed",
        ["deepspeed_configs/zero3_bf16.json", "deepspeed_configs/zero2.json"],
    )
    def test_deepspeed(self, temp_dir, deepspeed):
        pytest.importorskip("deepspeed")
        _run(
            temp_dir, _base_cfg(temp_dir) | {"deepspeed": str(AXOLOTL_ROOT / deepspeed)}
        )
