"""Module for testing prompt tokenizers."""
import json
import logging
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from axolotl.prompt_strategies.alpaca_chat import NoSystemPrompter
from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    ShareGPTPromptTokenizingStrategy,
)
from axolotl.prompters import AlpacaPrompter, ShareGPTPrompter

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
        prompter = ShareGPTPrompter("chat")
        strat = ShareGPTPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        example = strat.tokenize_prompt(conversation)
        for fields in ["input_ids", "attention_mask", "labels"]:
            self.assertEqual(len(example[fields]), len(tokenized_conversation[fields]))
            self.assertEqual(example[fields], tokenized_conversation[fields])

    def test_completion(self):
        """
        tests the interface between the user and assistant parts
        """
        prompter = NoSystemPrompter()
        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        sample = {
            "instruction": "hello cruel. lorem ipsum dolor sit amet.",
            "output": "world!",
        }
        example = strat.tokenize_prompt(sample)
        world_idx = example["input_ids"].index(3186)
        assert example["labels"][world_idx] == 3186
        assert example["labels"][world_idx - 1] == -100

    def test_alpaca(self):
        """
        tests the interface between the user and assistant parts
        """
        prompter = AlpacaPrompter()
        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        sample = {"instruction": "hello!", "output": "Hi! How can I help?"}
        example = strat.tokenize_prompt(sample)
        world_idx = example["input_ids"].index(6324)
        assert example["labels"][world_idx] == 6324
        assert example["labels"][world_idx - 1] == -100


if __name__ == "__main__":
    unittest.main()
