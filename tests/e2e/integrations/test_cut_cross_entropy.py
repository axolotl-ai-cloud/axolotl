"""
Simple end-to-end test for Cut Cross Entropy integration
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils import get_pytorch_version
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import (
    check_model_output_exists,
    check_tensorboard_loss_decreased,
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

    def test_llama_w_cce(self, min_cfg, temp_dir):
        cfg = DictDefault(min_cfg)
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
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
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
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
            "flash_attention",
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
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
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
