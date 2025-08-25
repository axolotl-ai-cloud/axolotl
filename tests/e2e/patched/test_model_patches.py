"""
E2E smoke tests to check that the monkeypatches are in place for certain configurations
"""

import unittest

import transformers

from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

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
                "val_set_size": 0.02,
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
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        tokenizer = load_tokenizer(cfg)
        ModelLoader(cfg, tokenizer, inference=False).load()

    @with_temp_dir
    def test_mistral_multipack(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "trl-internal-testing/tiny-MistralForCausalLM-0.2",
                "flash_attention": True,
                "sample_packing": True,
                "sequence_len": 2048,
                "val_set_size": 0.02,
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
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        tokenizer = load_tokenizer(cfg)
        ModelLoader(cfg, tokenizer, inference=False).load()

        assert (
            "torch.jit"
            in transformers.modeling_flash_attention_utils._get_unpad_data.__module__
        )
