"""
test module for the axolotl.utis.data module
"""
import unittest

from transformers import LlamaTokenizer

from axolotl.utils.data import encode_pretraining, md5


class TestEncodePretraining(unittest.TestCase):
    """
    test class for encode pretraining and md5 helper
    """

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
        result = encode_pretraining(self.tokenizer, self.max_tokens, examples["text"])

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


if __name__ == "__main__":
    unittest.main()
