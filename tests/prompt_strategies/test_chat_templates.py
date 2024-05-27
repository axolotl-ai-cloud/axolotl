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
    load,
)
from axolotl.utils.chat_templates import chat_templates
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="assistant_dataset")
def fixture_assistant_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                    },
                    {
                        "role": "assistant",
                        "content": "hello",
                    },
                    {
                        "role": "user",
                        "content": "goodbye",
                    },
                    {
                        "role": "assistant",
                        "content": "goodbye",
                    },
                ]
            }
        ]
    )


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


class TestAssistantChatTemplateLlama3:
    """
    Test class for assistant style datasets with llama-3 prompts using the chat_template strategy.
    """

    def test_llama3_load(self, llama3_tokenizer, assistant_dataset):
        # pylint: disable=duplicate-code
        strategy = load(
            llama3_tokenizer,
            DictDefault(
                {
                    "train_on_inputs": False,
                    "sequence_len": 512,
                }
            ),
            DictDefault(
                {
                    "chat_template": "llama3",
                    "message_field_role": "role",
                    "message_field_content": "content",
                    "roles": {
                        "user": ["user"],
                        "assistant": ["assistant"],
                        "system": ["system"],
                    },
                    "field_messages": "messages",
                }
            ),
        )
        res = strategy.tokenize_prompt(assistant_dataset[0])
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

    def test_llama3(self, llama3_tokenizer, assistant_dataset):
        # pylint: disable=duplicate-code
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                llama3_tokenizer,
                chat_templates("llama3"),
                message_field_role="role",
                message_field_content="content",
                roles={
                    "user": ["user"],
                    "assistant": ["assistant"],
                    "system": ["system"],
                },
            ),
            llama3_tokenizer,
            False,
            512,
        )
        strategy.messages = "messages"
        res = strategy.tokenize_prompt(assistant_dataset[0])
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
