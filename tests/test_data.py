"""
test module for the axolotl.utils.data module
"""

import unittest

from transformers import LlamaTokenizer

from axolotl.prompt_strategies.pretrain import load as load_pretrain
from axolotl.utils.data import encode_streaming, md5
from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import filter_sequences_by_length

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

    def _pretrain_strategy(self, sequence_len):
        cfg = DictDefault(
            {
                "train_on_inputs": False,
                "sequence_len": sequence_len,
                "pretraining_dataset": [{"text_column": "text"}],
            }
        )
        return load_pretrain(self.tokenizer, cfg)

    def test_long_document_is_chunked_not_dropped(self):
        """Long docs must be chunked into windows that survive the length filter (#3441)."""
        for sequence_len in (256, 512, 2048):
            with self.subTest(sequence_len=sequence_len):
                strat = self._pretrain_strategy(sequence_len)
                long_doc = " ".join(f"token{i}" for i in range(4 * sequence_len))
                windows = strat._tokenize(long_doc)["input_ids"]

                # the document spans more than one window (i.e. it was chunked)
                self.assertGreater(len(windows), 1)

                for window in windows:
                    # every window survives the downstream length filter ...
                    self.assertLessEqual(len(window), sequence_len)
                    self.assertTrue(
                        filter_sequences_by_length(
                            {"input_ids": window}, sequence_len=sequence_len
                        )
                    )
                    # ... and ends with EOS
                    self.assertEqual(window[-1], self.tokenizer.eos_token_id)

    def test_no_tokens_dropped_for_oversized_docs(self):
        """A doc longer than sequence_len must not be dropped entirely (#3441)."""
        sequence_len = 256
        strat = self._pretrain_strategy(sequence_len)
        long_doc = " ".join(f"token{i}" for i in range(2000))
        windows = strat._tokenize(long_doc)["input_ids"]

        kept = [
            w
            for w in windows
            if filter_sequences_by_length({"input_ids": w}, sequence_len=sequence_len)
        ]
        self.assertTrue(kept, "all windows were dropped — oversized doc lost entirely")
        self.assertEqual(len(kept), len(windows))

    def test_stride_below_window_size(self):
        """Tokenization must not raise from a stride >= effective max length."""
        for sequence_len in (256, 2048):
            with self.subTest(sequence_len=sequence_len):
                strat = self._pretrain_strategy(sequence_len)
                # would raise ValueError from the tokenizer if stride were too large
                strat._tokenize(" ".join(f"token{i}" for i in range(sequence_len * 3)))

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
        filter_sequences_by_length(data, 32, raise_on_drop=True)

        # This should return True, since data fits
        dropped = filter_sequences_by_length(data, 32)
        self.assertTrue(dropped)

        # This should raise
        self.assertRaises(
            ValueError, filter_sequences_by_length, data, 15, raise_on_drop=True
        )

        # This should return False, since data doesn't fit
        dropped = filter_sequences_by_length(data, 15)
        self.assertFalse(dropped)

        # -- batch sequence --
        # This should work
        data = {
            "input_ids": [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ]
        }
        filter_sequences_by_length(data, 32, raise_on_drop=True)

        # This should raise
        self.assertRaises(
            ValueError, filter_sequences_by_length, data, 15, raise_on_drop=True
        )

        # This should keep the first but drop the second entry
        dropped = filter_sequences_by_length(data, 15)
        self.assertEqual(dropped, [True, False])


if __name__ == "__main__":
    unittest.main()
