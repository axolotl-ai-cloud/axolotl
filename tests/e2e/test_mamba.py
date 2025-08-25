"""
E2E tests for lora llama
"""

import unittest

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


@pytest.mark.skip(reason="skipping until upstreamed into transformers")
class TestMamba(unittest.TestCase):
    """
    Test case for Mamba models
    """

    @with_temp_dir
    def test_fft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "state-spaces/mamba-130m",
                "model_type": "MambaLMHeadModel",
                "tokenizer_type": "AutoTokenizer",
                "tokenizer_config": "EleutherAI/gpt-neox-20b",
                "flash_attention": False,
                "sequence_len": 1024,
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
                "save_safetensors": False,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
