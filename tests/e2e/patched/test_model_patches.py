"""
E2E smoke tests to check that the monkeypatches are in place for certain configurations
"""

import unittest

import transformers

from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer

from ..utils import with_temp_dir


class TestModelPatches(unittest.TestCase):
    """
    TestCases for the multipack monkey patches
    """

    @with_temp_dir
    def test_mixtral_multipack(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "hf-internal-testing/Mixtral-tiny",
                "tokenizer_config": "LoneStriker/Mixtral-8x7B-v0.1-HF",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.1,
                "special_tokens": {},
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
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "eval_steps": 10,
            }
        )
        normalize_config(cfg)
        tokenizer = load_tokenizer(cfg)
        load_model(cfg, tokenizer, inference=False)

    @with_temp_dir
    def test_mistral_multipack(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "openaccess-ai-collective/tiny-mistral",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.1,
                "special_tokens": {},
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
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 20,
                "save_steps": 10,
                "eval_steps": 10,
            }
        )
        normalize_config(cfg)
        tokenizer = load_tokenizer(cfg)
        load_model(cfg, tokenizer, inference=False)

        assert (
            "torch.jit"
            in transformers.modeling_flash_attention_utils._get_unpad_data.__module__  # pylint: disable=protected-access
        )
