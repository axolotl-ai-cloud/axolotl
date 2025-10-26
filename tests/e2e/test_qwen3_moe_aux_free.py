"""
E2E smoke test for Aux-Loss-Free MoE routing on Qwen3-MoE tiny
"""

import unittest

import torch

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config, prepare_plugins
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestQwen3MoeAuxFree(unittest.TestCase):
    @with_temp_dir
    def test_qwen3_moe_aux_free_smoke(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
                "tokenizer_config": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
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

        # check that at least one sparse MoE block has been patched
        found = False
        for m in model.modules():
            if m.__class__.__name__.endswith("SparseMoeBlock") and hasattr(m, "_afb_patched"):
                assert m._afb_patched is True
                assert hasattr(m, "_afb_bias") and m._afb_bias.ndim == 1
                assert hasattr(m, "_afb_counts") and m._afb_counts.ndim == 1
                found = True
                break
        assert found, "No Qwen3-MoE sparse block patched by aux-free plugin"

        check_model_output_exists(temp_dir, cfg)
