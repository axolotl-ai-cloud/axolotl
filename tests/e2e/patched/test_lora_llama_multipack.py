"""
E2E tests for lora llama
"""

import logging
import os
import unittest

import pytest
from transformers.utils import is_auto_gptq_available, is_torch_bf16_gpu_available

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestLoraLlama(unittest.TestCase):
    """
    Test case for Llama models using LoRA w multipack
    """

    @with_temp_dir
    def test_lora_packing(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.2,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "max_steps": 20,
                "save_steps": 10,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @pytest.mark.skipif(not is_auto_gptq_available(), reason="auto-gptq not available")
    @with_temp_dir
    def test_lora_gptq_packed(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "TheBlokeAI/jackfram_llama-68m-GPTQ",
                "model_type": "AutoModelForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "gptq": True,
                "gptq_disable_exllama": True,
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.02,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "max_steps": 20,
                "save_steps": 0.5,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
