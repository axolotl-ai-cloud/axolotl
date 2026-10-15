"""
E2E tests for HF-format Mamba and Mamba2 with sample packing
"""

import unittest

import httpx
import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


def _packed_cfg(base_model, temp_dir, **overrides):
    return DictDefault(
        {
            "base_model": base_model,
            "flash_attention": False,
            "sequence_len": 1024,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            "load_in_8bit": False,
            "val_set_size": 0.0,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                },
            ],
            "gradient_checkpointing": False,
            "num_epochs": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "output_dir": temp_dir,
            "learning_rate": 0.00001,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "cosine",
            "max_steps": 2,
            "save_steps": 1,
            "eval_steps": None,
            "save_first_step": False,
            **overrides,
        }
    )


def _train(cfg):
    cfg = validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    train(cfg=cfg, dataset_meta=dataset_meta)
    check_model_output_exists(cfg.output_dir, cfg)


def _hub_mamba_kernels_available() -> bool:
    """Whether the hub kernels transformers maps Mamba2 onto have a build for this torch."""
    from kernels import has_kernel

    try:
        return has_kernel("kernels-community/mamba-ssm", version=2)
    except httpx.HTTPError:
        return False


class TestMamba(unittest.TestCase):
    """
    Test case for Mamba1 models on packed sequences; needs no kernels
    """

    @with_temp_dir
    def test_fft_packed(self, temp_dir):
        _train(_packed_cfg("state-spaces/mamba-130m-hf", temp_dir))


@pytest.mark.skipif(
    not _hub_mamba_kernels_available(),
    reason="no kernels-community/mamba-ssm build for this torch/CUDA",
)
class TestMamba2(unittest.TestCase):
    """
    Test case for Mamba2 models on packed sequences via the hub kernels
    """

    @with_temp_dir
    def test_fft_packed(self, temp_dir):
        _train(_packed_cfg("AntonV/mamba2-130m-hf", temp_dir, use_kernels=True))
