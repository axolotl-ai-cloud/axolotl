"""
Simple end-to-end test for Cut Cross Entropy integration
"""

from pathlib import Path

import pytest
from safetensors.torch import load_file

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils import get_pytorch_version
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import (
    check_model_output_exists,
    check_tensorboard_loss_decreased,
    requires_flash_attn,
)


@pytest.fixture()
def min_cfg(temp_dir):
    return {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "plugins": [
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],
        "cut_cross_entropy": True,
        "sequence_len": 1024,
        "val_set_size": 0.02,
        "special_tokens": {
            "pad_token": "<|endoftext|>",
        },
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            },
        ],
        "num_epochs": 1,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-4,
        "optimizer": "adamw_torch_fused",
        "output_dir": temp_dir,
        "lr_scheduler": "cosine",
        "max_steps": 40,
        "warmup_steps": 5,
        "bf16": "auto",
        "save_first_step": False,
        "use_tensorboard": True,
        "seed": 42,
    }


class TestCutCrossEntropyIntegration:
    """
    e2e tests for cut_cross_entropy integration with Axolotl
    """

    @pytest.mark.parametrize(
        "cce_overrides",
        [
            {},
            {"cut_cross_entropy_accum_c_fp32": True},
            {
                "cut_cross_entropy_accum_c_fp32": True,
                "cut_cross_entropy_c_grad_chunk_size": "auto",
            },
        ],
        ids=["default", "accum_c_fp32", "chunked_auto"],
    )
    def test_llama_w_cce(self, min_cfg, temp_dir, cce_overrides):
        cfg = DictDefault(min_cfg | cce_overrides)
        # plugin args only merge into the schema once the plugin is registered
        prepare_plugins(cfg)
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            check_tensorboard_loss_decreased(
                temp_dir + "/runs",
                initial_window=5,
                final_window=5,
                max_initial=2.2,
                max_final=2.0,
            )

    def test_qwen2_w_cce(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "plugins": [
                    "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
                ],
                "cut_cross_entropy": True,
                "sequence_len": 1024,
                "val_set_size": 0.02,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-4,
                "optimizer": "adamw_torch_fused",
                "output_dir": temp_dir,
                "lr_scheduler": "cosine",
                "max_steps": 50,
                "bf16": "auto",
                "save_first_step": False,
                "use_tensorboard": True,
                "seed": 42,
            }
        )
        prepare_plugins(cfg)
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            check_tensorboard_loss_decreased(
                temp_dir + "/runs",
                initial_window=5,
                final_window=5,
                max_initial=5.0,
                max_final=4.7,
            )

    @pytest.mark.parametrize(
        "attention_type",
        [
            pytest.param("flash_attention", marks=requires_flash_attn),
            "sdp_attention",
            # "xformers_attention",
        ],
    )
    def test_llama_w_cce_and_attention(self, min_cfg, temp_dir, attention_type):
        cfg = DictDefault(
            min_cfg
            | {
                attention_type: True,
            }
        )
        prepare_plugins(cfg)
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            check_tensorboard_loss_decreased(
                temp_dir + "/runs",
                initial_window=5,
                final_window=5,
                max_initial=2.2,
                max_final=2.0,
            )

    def test_llama_lora_lm_head_w_cce(self, min_cfg, temp_dir):
        cfg = DictDefault(
            min_cfg
            | {
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj", "v_proj", "lm_head"],
                "max_steps": 10,
            }
        )
        prepare_plugins(cfg)
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        # lora_B starts at zero, so a head adapter CCE never saw would be saved as all zeros.
        adapter = load_file(str(Path(temp_dir) / "adapter_model.safetensors"))
        lm_head_b = [v for k, v in adapter.items() if "lm_head" in k and "lora_B" in k]
        assert lm_head_b, list(adapter)
        assert all(t.float().abs().sum() > 0 for t in lm_head_b)
