"""
E2E smoke test for Llama 4 aux-loss-free routing via plugin
"""

import unittest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestLlama4MoeAuxFree(unittest.TestCase):
    """Smoke test to ensure aux-free plugin patches Llama 4 MoE correctly."""

    @with_temp_dir
    def test_llama4_aux_free_smoke(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "yujiepan/llama-4-tiny-random",
                "tokenizer_config": "yujiepan/llama-4-tiny-random",
                "trust_remote_code": True,
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

        patched = next((m for m in model.modules() if hasattr(m, "_afb_bias")), None)
        assert patched is not None, "Llama 4 MoE layer was not patched by aux-free plugin"
        assert patched._afb_bias.ndim == 1
        assert patched._afb_counts.ndim == 1
        check_model_output_exists(temp_dir, cfg)


if __name__ == "__main__":
    unittest.main()
