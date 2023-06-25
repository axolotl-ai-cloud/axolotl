"""
Test cases for the tokenizer loading
"""
import unittest

from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_tokenizer


class TestTokenizers(unittest.TestCase):
    """
    test class for the load_tokenizer fn
    """

    def test_default_use_fast(self):
        cfg = DictDefault({})
        tokenizer = load_tokenizer("huggyllama/llama-7b", None, cfg)
        assert "Fast" in tokenizer.__class__.__name__

    def test_dont_use_fast(self):
        cfg = DictDefault(
            {
                "tokenizer_use_fast": False,
            }
        )
        tokenizer = load_tokenizer("huggyllama/llama-7b", None, cfg)
        assert "Fast" not in tokenizer.__class__.__name__


if __name__ == "__main__":
    unittest.main()
