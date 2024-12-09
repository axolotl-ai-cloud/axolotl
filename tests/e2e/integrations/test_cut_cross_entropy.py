"""
Simple end-to-end test for Cut Cross Entropy integration
"""

from pathlib import Path

import pytest

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils import get_pytorch_version
from axolotl.utils.config import normalize_config, prepare_plugins
from axolotl.utils.dict import DictDefault

# pylint: disable=duplicate-code


@pytest.fixture()
def min_cfg(temp_dir):
    return {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "plugins": [
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],
        "cut_cross_entropy": True,
        "sequence_len": 1024,
        "val_set_size": 0.1,
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
        "learning_rate": 0.00001,
        "optimizer": "adamw_torch",
        "output_dir": temp_dir,
        "lr_scheduler": "cosine",
        "save_safetensors": True,
        "max_steps": 10,
        "bf16": "auto",
    }


class TestCutCrossEntropyIntegration:
    """
    e2e tests for cut_cross_entropy integration with Axolotl
    """

    # pylint: disable=redefined-outer-name
    def test_llama_w_cce(self, min_cfg, temp_dir):
        cfg = DictDefault(min_cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
            assert (Path(temp_dir) / "model.safetensors").exists()

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
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            with pytest.raises(ImportError):
                train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        else:
            train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
            assert (Path(temp_dir) / "model.safetensors").exists()
