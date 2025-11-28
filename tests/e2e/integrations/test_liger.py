"""
Simple end-to-end test for Liger integration
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists, require_torch_2_4_1


class LigerIntegrationTestCase:
    """
    e2e tests for liger integration with Axolotl
    """

    @require_torch_2_4_1
    def test_llama_wo_flce(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "plugins": [
                    "axolotl.integrations.liger.LigerPlugin",
                ],
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_glu_activation": True,
                "liger_cross_entropy": True,
                "liger_fused_linear_cross_entropy": False,
                "sequence_len": 1024,
                "val_set_size": 0.05,
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
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 5,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @require_torch_2_4_1
    @pytest.mark.parametrize(
        "liger_use_token_scaling",
        [True, False],
    )
    def test_llama_w_flce(self, temp_dir, liger_use_token_scaling):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "plugins": [
                    "axolotl.integrations.liger.LigerPlugin",
                ],
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_glu_activation": True,
                "liger_cross_entropy": False,
                "liger_fused_linear_cross_entropy": True,
                "liger_use_token_scaling": liger_use_token_scaling,
                "sequence_len": 1024,
                "val_set_size": 0.05,
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
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 5,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
