"""
E2E smoke tests for Aux-Loss-Free MoE routing via plugin
"""

import unittest

import torch

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config, prepare_plugins
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestMoeAuxFree(unittest.TestCase):
    """Smoke tests to ensure aux-free plugin enables and runs on Mixtral tiny."""

    @with_temp_dir
    def test_mixtral_aux_free_smoke(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": False,
                "sequence_len": 512,
                "bf16": False,
                "fp16": False,
                "val_set_size": 0.02,
                "special_tokens": {},
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-5,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_steps": 0,
                "eval_steps": 0,
                "save_first_step": False,
                # Aux-free plugin and toggles
                "plugins": [
                    "axolotl.integrations.aux_free_router.plugin.AuxFreeMoEPlugin",
                ],
                "moe_balance_type": "noaux_tc",
                "moe_update_rate": 0.01,
                "moe_update_momentum": 0.9,
                "moe_bias_cap": 2.0,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        prepare_plugins(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        model, _, _ = train(cfg=cfg, dataset_meta=dataset_meta)

        # Inspect model modules for a patched MoE layer
        patched = None
        for m in model.modules():
            if hasattr(m, "_afb_patched") and getattr(m, "_afb_patched") is True:
                patched = m
                break
        assert patched is not None, "No MoE layer patched by aux-free plugin"
        assert hasattr(patched, "_afb_bias") and patched._afb_bias.ndim == 1
        assert hasattr(patched, "_afb_counts") and patched._afb_counts.ndim == 1
        # ensure counts buffer got reset by callback (best effort)
        assert torch.all(patched._afb_counts == 0)

        check_model_output_exists(temp_dir, cfg)
