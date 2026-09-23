"""
E2E tests for HF-format Mamba2 with sample packing
"""

import unittest

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


def _hub_mamba_kernels_available() -> bool:
    """Whether the hub kernels transformers maps Mamba2 onto have a build for this torch."""
    try:
        from kernels import get_kernel

        get_kernel("kernels-community/mamba-ssm", version=2)
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return True


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
        cfg = DictDefault(
            {
                "base_model": "AntonV/mamba2-130m-hf",
                "flash_attention": False,
                "use_kernels": True,
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
                "max_steps": 20,
                "save_steps": 10,
                "eval_steps": None,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
