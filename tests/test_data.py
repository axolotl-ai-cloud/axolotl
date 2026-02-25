"""
test module for the axolotl.utils.data module
"""

import unittest

from datasets import Dataset
from transformers import LlamaTokenizer

from axolotl.utils.data import encode_streaming, md5
from axolotl.utils.data.streaming import _chunk_long_sequences
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


class TestChunkLongSequences(unittest.TestCase):
    """Tests for the _chunk_long_sequences helper function."""

    def test_no_long_sequences(self):
        """Early return when no sequences exceed max_seq_length."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "attention_mask": [[1, 1, 1], [1, 1, 1]],
                "labels": [[1, 2, 3], [4, 5, 6]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=5)
        # Should return the same data unchanged
        self.assertEqual(len(result["input_ids"]), 2)
        self.assertEqual(result["input_ids"][0], [1, 2, 3])
        self.assertEqual(result["input_ids"][1], [4, 5, 6])

    def test_sequence_exactly_at_max(self):
        """Sequences exactly at max_seq_length should not be chunked."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1, 1]],
                "labels": [[1, 2, 3, 4, 5]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=5)
        self.assertEqual(len(result["input_ids"]), 1)
        self.assertEqual(result["input_ids"][0], [1, 2, 3, 4, 5])

    def test_sequence_requires_multiple_chunks(self):
        """A long sequence should be split into max_seq_length-sized chunks."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                "labels": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=4)
        # 10 tokens / 4 = 3 chunks (4, 4, 2)
        self.assertEqual(len(result["input_ids"]), 3)
        self.assertEqual(result["input_ids"][0], [1, 2, 3, 4])
        self.assertEqual(result["input_ids"][1], [5, 6, 7, 8])
        self.assertEqual(result["input_ids"][2], [9, 10])
        # Check attention_mask and labels are chunked the same way
        self.assertEqual(result["attention_mask"][0], [1, 1, 1, 1])
        self.assertEqual(result["labels"][2], [9, 10])

    def test_trailing_short_chunk(self):
        """The last chunk of a split can be shorter than max_seq_length."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4, 5, 6, 7]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=3)
        # 7 tokens / 3 = 3 chunks (3, 3, 1)
        self.assertEqual(len(result["input_ids"]), 3)
        self.assertEqual(result["input_ids"][0], [1, 2, 3])
        self.assertEqual(result["input_ids"][1], [4, 5, 6])
        self.assertEqual(result["input_ids"][2], [7])

    def test_mixed_short_and_long(self):
        """Mix of short and long sequences; only long ones are chunked."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2], [3, 4, 5, 6, 7, 8], [9, 10]],
                "attention_mask": [[1, 1], [1, 1, 1, 1, 1, 1], [1, 1]],
                "labels": [[1, 2], [3, 4, 5, 6, 7, 8], [9, 10]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=4)
        # First sample (2 tokens): kept as is
        # Second sample (6 tokens): split into 2 chunks (4, 2)
        # Third sample (2 tokens): kept as is
        self.assertEqual(len(result["input_ids"]), 4)
        self.assertEqual(result["input_ids"][0], [1, 2])
        self.assertEqual(result["input_ids"][1], [3, 4, 5, 6])
        self.assertEqual(result["input_ids"][2], [7, 8])
        self.assertEqual(result["input_ids"][3], [9, 10])

    def test_without_labels(self):
        """Chunking should work when labels column is absent."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4, 5, 6]],
                "attention_mask": [[1, 1, 1, 1, 1, 1]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=4)
        self.assertEqual(len(result["input_ids"]), 2)
        self.assertEqual(result["input_ids"][0], [1, 2, 3, 4])
        self.assertEqual(result["input_ids"][1], [5, 6])
        self.assertNotIn("labels", result)

    def test_without_attention_mask(self):
        """Chunking should work when attention_mask column is absent."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4, 5, 6]],
                "labels": [[1, 2, 3, 4, 5, 6]],
            }
        )
        result = _chunk_long_sequences(ds, max_seq_length=4)
        self.assertEqual(len(result["input_ids"]), 2)
        self.assertEqual(result["input_ids"][0], [1, 2, 3, 4])
        self.assertEqual(result["labels"][1], [5, 6])
        self.assertNotIn("attention_mask", result)


if __name__ == "__main__":
    unittest.main()
