"""
Test cases for the tokenizer loading
"""
import unittest

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_tokenizer


class TestTokenizers(unittest.TestCase):
    """
    test class for the load_tokenizer fn
    """

    def test_default_use_fast(self):
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert "Fast" in tokenizer.__class__.__name__

    def test_dont_use_fast(self):
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "tokenizer_use_fast": False,
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert "Fast" not in tokenizer.__class__.__name__

    def test_special_tokens_modules_to_save(self):
        # setting special_tokens to new token
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "adapter": "lora",
                "special_tokens": {"bos_token": "[INST]"},
            }
        )
        with pytest.raises(
            ValueError,
            match=r".*Please set lora_modules_to_save*",
        ):
            load_tokenizer(cfg)

        # setting special_tokens but not changing from default
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "adapter": "lora",
                "special_tokens": {"bos_token": "<s>"},
            }
        )
        load_tokenizer(cfg)

        # non-adapter setting special_tokens
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "special_tokens": {"bos_token": "[INST]"},
            }
        )
        load_tokenizer(cfg)

    def test_add_additional_special_tokens(self):
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "special_tokens": {"additional_special_tokens": ["<|im_start|>"]},
            }
        )
        tokenizer = load_tokenizer(cfg)
        self.assertEqual(tokenizer("<|im_start|>user")["input_ids"], [1, 32000, 1404])
        self.assertEqual(len(tokenizer), 32001)

        # ensure reloading the tokenizer again from cfg results in same vocab length
        tokenizer = load_tokenizer(cfg)
        self.assertEqual(len(tokenizer), 32001)


if __name__ == "__main__":
    unittest.main()
