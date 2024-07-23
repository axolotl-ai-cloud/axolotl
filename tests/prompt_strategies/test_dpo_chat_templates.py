"""
tests for chat_template prompt strategy
"""

import unittest

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.prompt_strategies.dpo.chat_template import default
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
                ],
                "chosen": {
                    "role": "assistant",
                    "content": "goodbye",
                },
                "rejected": {
                    "role": "assistant",
                    "content": "party on",
                },
            }
        ]
    )


@pytest.fixture(name="custom_assistant_dataset")
def fixture_custom_assistant_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {
                        "speaker": "human",
                        "text": "hello",
                    },
                    {
                        "speaker": "agent",
                        "text": "hello",
                    },
                    {
                        "speaker": "human",
                        "text": "goodbye",
                    },
                ],
                "better": {
                    "speaker": "agent",
                    "text": "goodbye",
                },
                "worse": {
                    "speaker": "agent",
                    "text": "party on",
                },
            }
        ]
    )


@pytest.fixture(name="llama3_tokenizer")
def fixture_llama3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    tokenizer.eos_token = "<|eot_id|>"

    return tokenizer


class TestAssistantDPOChatTemplateLlama3:
    """
    Test class for assistant style datasets with llama-3 prompts using the chat_template strategy.
    """

    def test_llama3_defaults(self, llama3_tokenizer, assistant_dataset):
        # pylint: disable=duplicate-code
        transform_fn = default(
            DictDefault(
                {
                    "chat_template": "llama3",
                    "datasets": [
                        {
                            "chat_template": "llama3",
                        }
                    ],
                }
            )
        )
        result = transform_fn(assistant_dataset[0], tokenizer=llama3_tokenizer)
        assert result["prompt"] == (
            "<|begin_of_text|>"
            + "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n\ngoodbye<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert result["chosen"] == "goodbye<|eot_id|>"
        assert result["rejected"] == "party on<|eot_id|>"

    def test_llama3_configured(self, llama3_tokenizer, custom_assistant_dataset):
        # pylint: disable=duplicate-code
        transform_fn = default(
            DictDefault(
                {
                    "chat_template": "llama3",
                    "datasets": [
                        {
                            "chat_template": "llama3",
                            "field_messages": "conversation",
                            "field_chosen": "better",
                            "field_rejected": "worse",
                            "message_field_role": "speaker",
                            "message_field_content": "text",
                            "roles": {
                                "user": ["human"],
                                "assistant": ["agent"],
                                "system": ["sys"],
                            },
                        }
                    ],
                }
            )
        )
        result = transform_fn(custom_assistant_dataset[0], tokenizer=llama3_tokenizer)
        assert result["prompt"] == (
            "<|begin_of_text|>"
            + "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n\ngoodbye<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert result["chosen"] == "goodbye<|eot_id|>"
        assert result["rejected"] == "party on<|eot_id|>"


if __name__ == "__main__":
    unittest.main()
