"""
tests for chat_template prompt strategy
"""

import unittest

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.prompt_strategies.dpo.chat_template import default
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline


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


@pytest.fixture(name="phi3_tokenizer")
@enable_hf_offline
def fixture_phi3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-128k-instruct")

    return tokenizer


@pytest.fixture(name="gemma_tokenizer")
@enable_hf_offline
def fixture_gemma_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2b-it", revision="703fb4a")

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
                            "type": "chat_template",
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
                            "type": "chat_template",
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


class TestAssistantDPOChatTemplatePhi3:
    """
    Test class for assistant style datasets with phi-3 prompts using the tokenizer's chat_template strategy.
    """

    def test_phi3_defaults(self, phi3_tokenizer, assistant_dataset):
        # pylint: disable=duplicate-code
        transform_fn = default(
            DictDefault(
                {
                    "chat_template": "tokenizer_default",
                    "datasets": [
                        {
                            "type": "chat_template",
                        }
                    ],
                }
            )
        )
        result = transform_fn(assistant_dataset[0], tokenizer=phi3_tokenizer)
        assert result["prompt"] == (
            "<|user|>\nhello<|end|>\n"
            + "<|assistant|>\nhello<|end|>\n"
            + "<|user|>\ngoodbye<|end|>\n"
            + "<|assistant|>\n"
        )
        assert result["chosen"] == "goodbye<|end|>"
        assert result["rejected"] == "party on<|end|>"


class TestAssistantDPOChatTemplateGemma:
    """
    Test class for assistant style datasets with gemma prompts using the tokenizer's chat_template strategy.
    """

    def test_gemma_defaults(self, gemma_tokenizer, assistant_dataset):
        # pylint: disable=duplicate-code
        transform_fn = default(
            DictDefault(
                {
                    "chat_template": "tokenizer_default",
                    "datasets": [
                        {
                            "type": "chat_template",
                        }
                    ],
                }
            )
        )
        result = transform_fn(assistant_dataset[0], tokenizer=gemma_tokenizer)
        assert result["prompt"] == (
            "<bos><start_of_turn>user\nhello<end_of_turn>\n"
            + "<start_of_turn>model\nhello<end_of_turn>\n"
            + "<start_of_turn>user\ngoodbye<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )
        assert result["chosen"] == "goodbye<end_of_turn>"
        assert result["rejected"] == "party on<end_of_turn>"


if __name__ == "__main__":
    unittest.main()
