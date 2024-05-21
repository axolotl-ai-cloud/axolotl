"""
tests for chat_template prompt strategy
"""
import unittest

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.utils.chat_templates import chat_templates


@pytest.fixture(name="sharegpt_dataset")
def fixture_sharegpt_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "hello",
                    },
                    {
                        "from": "gpt",
                        "value": "hello",
                    },
                    {
                        "from": "human",
                        "value": "goodbye",
                    },
                    {
                        "from": "gpt",
                        "value": "goodbye",
                    },
                ]
            }
        ]
    )


@pytest.fixture(name="llama3_tokenizer")
def fixture_llama3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    tokenizer.eos_token = "<|eot_id|>"

    return tokenizer


class TestSharegptChatTemplateLlama3:
    """
    Test class for ShareGPT style datasets with llama-3 prompts using the chat_template strategy.
    """

    def test_llama3(self, llama3_tokenizer, sharegpt_dataset):
        # pylint: disable=duplicate-code
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            llama3_tokenizer,
            False,
            512,
        )
        res = strategy.tokenize_prompt(sharegpt_dataset[0])
        input_ids = res["input_ids"]
        # fmt: off
        assert input_ids == [
            128000,  # bos
            128006, 882, 128007,  # user header
            271, 15339, 128009,  # user prompt eot
            128006, 78191, 128007,  # assistant header
            271, 15339, 128009,   # assistant response eot
            128006, 882, 128007,
            271, 19045, 29474, 128009,
            128006, 78191, 128007,
            271, 19045, 29474, 128009,
        ]
        # fmt: on


if __name__ == "__main__":
    unittest.main()
