"""
test module for the axolotl.utils.data module
"""

import unittest

from transformers import LlamaTokenizer

from axolotl.utils.data import encode_streaming, md5
from axolotl.utils.trainer import drop_long_seq

from tests.hf_offline_utils import enable_hf_offline


class TestEncodePretraining(unittest.TestCase):
    """
    test class for encode pretraining and md5 helper
    """

    @enable_hf_offline
    def setUp(self):
        self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        self.max_tokens = 15  # set a small number for easy inspection

    def test_encode_pretraining(self):
        examples = {
            "text": [
                "Hello, world!",
                "Nice to meet you.",
                "lorem ipsum dolor sit amet.",
                "Nice to meet you again!.",
                "hello, hello",
            ]
        }
        result = encode_streaming(examples, self.tokenizer, self.max_tokens)

        self.assertEqual(len(result["input_ids"]), 3)

        # Assert the length of input_ids and attention_mask is correct
        self.assertEqual(len(result["input_ids"][0]), self.max_tokens)
        self.assertEqual(len(result["attention_mask"][0]), self.max_tokens)

        # Assert EOS and PAD tokens are correctly added
        # hello world! is 4 tokens
        self.assertEqual(result["input_ids"][0][0], self.tokenizer.bos_token_id)
        self.assertEqual(result["input_ids"][0][5], self.tokenizer.eos_token_id)
        self.assertEqual(result["input_ids"][0][6], self.tokenizer.pad_token_id)
        # second part, 5 tokens
        self.assertEqual(result["input_ids"][0][7], self.tokenizer.bos_token_id)
        self.assertEqual(result["input_ids"][0][13], self.tokenizer.eos_token_id)
        self.assertEqual(result["input_ids"][0][14], self.tokenizer.pad_token_id)

    def test_md5(self):
        self.assertEqual(md5("hello world"), "5eb63bbbe01eeed093cb22bb8f5acdc3")
        self.assertEqual(
            md5("hello world", "utf-8"), "5eb63bbbe01eeed093cb22bb8f5acdc3"
        )

    def test_excess_length_strategy(self):
        """Test that excess_length_strategy results in a value error when set to 'raise'."""

        # -- single sequence --
        # This should work
        data = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
        drop_long_seq(data, 32, raise_on_drop=True)

        # This should return True, since data fits
        dropped = drop_long_seq(data, 32)
        self.assertTrue(dropped)

        # This should raise
        self.assertRaises(ValueError, drop_long_seq, data, 15, raise_on_drop=True)

        # This should return False, since data doesn't fit
        dropped = drop_long_seq(data, 15)
        self.assertFalse(dropped)

        # -- batch sequence --
        # This should work
        data = {
            "input_ids": [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ]
        }
        drop_long_seq(data, 32, raise_on_drop=True)

        # This should raise
        self.assertRaises(ValueError, drop_long_seq, data, 15, raise_on_drop=True)

        # This should keep the first but drop the second entry
        dropped = drop_long_seq(data, 15)
        self.assertEqual(dropped, [True, False])


if __name__ == "__main__":
    unittest.main()
