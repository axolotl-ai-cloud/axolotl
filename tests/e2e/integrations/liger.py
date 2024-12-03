"""
Simple end-to-end test for Liger integration
"""
import unittest
from pathlib import Path

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins
from axolotl.utils.dict import DictDefault

from ..utils import with_temp_dir


class LigerIntegrationTestCase(unittest.TestCase):
    """
    e2e tests for liger integration with Axolotl
    """

    @with_temp_dir
    def test_llama_wo_flce(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "plugins": [
                    "axolotl.integrations.liger.LigerPlugin",
                ],
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_swiglu": True,
                "liger_cross_entropy": True,
                "liger_fused_linear_cross_entropy": False,
                "sequence_len": 1024,
                "val_set_size": 0.1,
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
                "num_epochs": 1,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 10,
            }
        )
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()

    @with_temp_dir
    def test_llama_w_flce(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "plugins": [
                    "axolotl.integrations.liger.LigerPlugin",
                ],
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_swiglu": True,
                "liger_cross_entropy": False,
                "liger_fused_linear_cross_entropy": True,
                "sequence_len": 1024,
                "val_set_size": 0.1,
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
                "num_epochs": 1,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 10,
            }
        )
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()
