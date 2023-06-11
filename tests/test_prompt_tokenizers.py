"""Module for testing prompt tokenizers."""
import json
import logging
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from rathe import ChatPromptFormatter, ShareGPTParser, TokenizationOptions
from rathe.pipeline import DataPipeline

logging.basicConfig(level="INFO")


class TestPromptTokenizationStrategies(unittest.TestCase):
    """
    Test class for prompt tokenization strategies.
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )

    def test_sharegpt_integration(self):
        print(Path(__file__).parent)
        with open(
            Path(__file__).parent / "fixtures/conversation.json", encoding="utf-8"
        ) as fin:
            data = fin.read()
            conversation = json.loads(data)
        with open(
            Path(__file__).parent / "fixtures/conversation.tokenized.json",
            encoding="utf-8",
        ) as fin:
            data = fin.read()
            tokenized_conversation = json.loads(data)

        parser = ShareGPTParser()
        formatter = ChatPromptFormatter.vicuna()
        formatter.system_prompt = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. '
        formatter.user_wrapper.suffix = ''
        formatter.model_wrapper.suffix = '</s>'
        pipeline = DataPipeline(parser, formatter, self.tokenizer)

        example = pipeline(conversation)
        for fields in ["input_ids", "attention_mask", "labels"]:
            self.assertEqual(len(example[fields]), len(tokenized_conversation[fields]))
            self.assertEqual(example[fields], tokenized_conversation[fields])


if __name__ == "__main__":
    unittest.main()
