"""
Parity test comparing aux-loss (gshard) vs aux-loss-free (noaux_tc) on Mixtral-tiny.
Checks that aux-free training loss does not degrade beyond a small tolerance.
"""

import unittest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config, prepare_plugins
from axolotl.utils.dict import DictDefault

from .utils import with_temp_dir


def _last_logged_loss(trainer) -> float | None:
    # Scan log_history for the most recent entry with a 'loss' key
    for entry in reversed(trainer.state.log_history):
        if isinstance(entry, dict) and "loss" in entry:
            return float(entry["loss"])
    return None


class TestMoeAuxParity(unittest.TestCase):
    @with_temp_dir
    def test_mixtral_auxfree_vs_auxloss_loss_parity(self, temp_dir):
        base_cfg = {
            "base_model": "hf-internal-testing/Mixtral-tiny",
            "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
            "flash_attention": False,
            "sequence_len": 512,
            "bf16": False,
            "fp16": False,
            "val_set_size": 0.02,
            "special_tokens": {},
            "datasets": [
                {"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"},
            ],
            "num_epochs": 1,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "max_steps": 8,
            "save_steps": 0,
            "eval_steps": 0,
            "save_first_step": False,
            "seed": 42,
            "logging_steps": 1,
        }

        # Baseline: aux-loss (gshard)
        cfg0 = DictDefault(dict(base_cfg))
        cfg0.output_dir = f"{temp_dir}/baseline"
        cfg0 = validate_config(cfg0)
        normalize_config(cfg0)
        # baseline uses default aux-loss routing; no plugin registration
        dataset_meta0 = load_datasets(cfg=cfg0)
        model0, _, trainer0 = train(cfg=cfg0, dataset_meta=dataset_meta0)
        loss0 = _last_logged_loss(trainer0)
        assert loss0 is not None

        # Aux-free: plugin + noaux_tc
        cfg1 = DictDefault(dict(base_cfg))
        cfg1.output_dir = f"{temp_dir}/auxfree"
        cfg1.plugins = [
            "axolotl.integrations.aux_free_router.plugin.AuxFreeMoEPlugin",
        ]
        cfg1.moe_balance_type = "noaux_tc"
        cfg1.moe_update_rate = 0.01
        cfg1.moe_update_momentum = 0.9
        cfg1.moe_bias_cap = 2.0
        cfg1 = validate_config(cfg1)
        normalize_config(cfg1)
        prepare_plugins(cfg1)
        dataset_meta1 = load_datasets(cfg=cfg1)
        model1, _, trainer1 = train(cfg=cfg1, dataset_meta=dataset_meta1)
        loss1 = _last_logged_loss(trainer1)
        assert loss1 is not None

        # Assert aux-free loss is within 10% of aux-loss baseline
        assert loss1 <= 1.1 * loss0, f"aux-free loss {loss1} > 1.1 * baseline {loss0}"
