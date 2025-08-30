"""
E2E tests for lora llama
"""

import unittest

import pytest
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, with_temp_dir


@pytest.mark.skip("FIXME, mostly underused functionality")
class TestFusedLlama(unittest.TestCase):
    """
    Test case for Llama models using Fused layers
    """

    @with_temp_dir
    def test_fft_packing(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "pad_to_sequence_len": True,
                "flash_attn_fuse_mlp": True,
                "sample_packing": True,
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
                "num_epochs": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 10,
                "save_steps": 5,
                "eval_steps": 5,
                "save_first_step": False,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
