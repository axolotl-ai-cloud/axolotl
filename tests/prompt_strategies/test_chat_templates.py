"""
tests for chat_template prompt strategy
"""

import logging
import unittest

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
    load,
)
from axolotl.prompters import IGNORE_TOKEN_ID
from axolotl.utils.chat_templates import chat_templates
from axolotl.utils.dict import DictDefault

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("axolotl")


@pytest.fixture(name="assistant_dataset")
def fixture_assistant_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "goodbye"},
                    {"role": "assistant", "content": "goodbye"},
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
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hello"},
                    {"from": "human", "value": "goodbye"},
                    {"from": "gpt", "value": "goodbye"},
                ]
            }
        ]
    )


@pytest.fixture(name="basic_dataset")
def fixture_basic_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "system", "value": "You are an AI assistant."},
                    {"from": "human", "value": "Hello"},
                    {"from": "assistant", "value": "Hi there!"},
                    {"from": "human", "value": "How are you?"},
                    {"from": "assistant", "value": "I'm doing well, thank you!"},
                ]
            }
        ]
    )


@pytest.fixture(name="llama3_tokenizer")
def fixture_llama3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

    return tokenizer


@pytest.fixture(name="phi35_tokenizer")
def fixture_phi35_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    return tokenizer


class TestAssistantChatTemplateLlama3:
    """
    Test class for assistant style datasets with llama-3 prompts using the chat_template strategy.
    """

    def test_llama3_load(self, llama3_tokenizer, assistant_dataset):
        LOG.info("Loading llama-3 tokenizer with assistant dataset")
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
        expected_input_ids = [
            128000,  # bos
            128006, 882, 128007,  # user header
            271, 15339, 128009,  # user prompt eot
            128006, 78191, 128007,  # assistant header
            271, 15339, 128009,  # assistant response eot
            128006, 882, 128007,
            271, 19045, 29474, 128009,
            128006, 78191, 128007,
            271, 19045, 29474, 128009,
        ]
        # fmt: on
        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"

    def test_llama3(self, llama3_tokenizer, assistant_dataset):
        LOG.info("Testing llama-3 with assistant dataset")
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
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
        )
        strategy.messages = "messages"
        res = strategy.tokenize_prompt(assistant_dataset[0])
        input_ids = res["input_ids"]
        # fmt: off
        expected_input_ids = [
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
        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"

    def test_phi35(self, phi35_tokenizer, assistant_dataset):
        LOG.info("Testing phi-3.5 with assistant dataset")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                phi35_tokenizer,
                chat_templates("phi_35"),
                message_field_role="role",
                message_field_content="content",
                roles={
                    "user": ["user"],
                    "assistant": ["assistant"],
                    "system": ["system"],
                },
            ),
            tokenizer=phi35_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
        )
        strategy.messages = "messages"
        res = strategy.tokenize_prompt(assistant_dataset[0])
        input_ids = res["input_ids"]
        labels = res["labels"]
        # fmt: off
        expected_input_ids = [
            32010,  # user
            22172, 32007,  # user eot
            32001,  # assistant
            22172, 32007,  # assistant eot
            32010,  # user
            1781, 26966, 32007,  # user eot
            32001,  # assistant
            1781, 26966, 32007,  # assistant eot
            32000,  # eos
        ]
        expected_labels = [
            -100,  # user
            -100, -100,  # user eot
            -100,  # assistant
            -100, -100,  # assistant eot,
            -100,  # user
            -100, -100, -100,  # user eot
            -100,  # assistant
            1781, 26966, 32007,  # assistant eot
            32000,  # eos
        ]
        # fmt: on
        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"

        LOG.debug(f"Expected labels : {expected_labels}")
        LOG.debug(f"Actual labels : {labels}")
        assert (
            labels == expected_labels
        ), f"Input IDs mismatch: {labels} != {expected_labels}"

    def test_llama3_with_training_data(self, llama3_tokenizer, assistant_dataset):
        LOG.info("Testing llama-3 with assistant dataset including training data")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                llama3_tokenizer,
                chat_templates("llama3"),
                message_field_role="role",
                message_field_content="content",
                message_field_training="training",
                roles={
                    "user": ["user"],
                    "assistant": ["assistant"],
                    "system": ["system"],
                },
            ),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            train_on_eos="none",
            sequence_len=512,
            roles_to_train=["assistant"],
        )
        strategy.messages = "messages"
        prompt_tokens = strategy.prompter.build_prompt(
            assistant_dataset[0]["messages"], False
        )
        prompt = llama3_tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        LOG.debug(f"Generated prompt: {prompt}")
        res = strategy.tokenize_prompt(assistant_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]
        # fmt: off
        expected_labels = [
            IGNORE_TOKEN_ID,  # bos
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # user header
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # user prompt eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # assistant header
            IGNORE_TOKEN_ID, 15339, IGNORE_TOKEN_ID,  # assistant response eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, 19045, 29474, IGNORE_TOKEN_ID,
        ]
        # fmt: on

        LOG.debug(f"Expected labels: {expected_labels}")
        LOG.debug(f"Actual labels: {labels}")
        assert labels == expected_labels, (
            f"Labels mismatch:\n"
            f"Expected: {expected_labels}\n"
            f"Actual: {labels}\n"
            f"Input IDs: {input_ids}\n"
        )


class TestSharegptChatTemplateLlama3:
    """
    Test class for ShareGPT style datasets with llama-3 prompts using the chat_template strategy.
    """

    def test_llama3_assistant(self, llama3_tokenizer, sharegpt_dataset):
        LOG.info("Testing ShareGPT style datasets with llama-3 assistant prompts")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            train_on_eos="none",
            sequence_len=512,
            roles_to_train=["gpt"],
        )
        res = strategy.tokenize_prompt(sharegpt_dataset[0])
        input_ids = res["input_ids"]
        labels = res["labels"]
        # fmt: off
        expected_input_ids = [
            128000,  # bos
            128006, 882, 128007,  # user header
            271, 15339, 128009,  # user prompt eot
            128006, 78191, 128007,  # assistant header
            271, 15339, 128009,  # assistant response eot
            128006, 882, 128007,
            271, 19045, 29474, 128009,
            128006, 78191, 128007,
            271, 19045, 29474, 128009,
        ]
        expected_labels = [
            IGNORE_TOKEN_ID,  # bos
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # user header
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # user prompt eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # assistant header
            IGNORE_TOKEN_ID, 15339, IGNORE_TOKEN_ID,  # assistant response eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, 19045, 29474, IGNORE_TOKEN_ID,
        ]
        # fmt: on

        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        LOG.debug(f"Expected labels: {expected_labels}")
        LOG.debug(f"Actual labels: {labels}")

        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"
        assert (
            labels == expected_labels
        ), f"Labels mismatch: {labels} != {expected_labels}"

    def test_llama3_human(self, llama3_tokenizer, sharegpt_dataset):
        LOG.info("Testing ShareGPT style datasets with llama-3 human prompts")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            train_on_eos="none",
            sequence_len=512,
            roles_to_train=["human"],
        )
        res = strategy.tokenize_prompt(sharegpt_dataset[0])
        input_ids = res["input_ids"]
        labels = res["labels"]
        # fmt: off
        expected_input_ids = [
            128000,  # bos
            128006, 882, 128007,  # user header
            271, 15339, 128009,  # user prompt eot
            128006, 78191, 128007,  # assistant header
            271, 15339, 128009,  # assistant response eot
            128006, 882, 128007,
            271, 19045, 29474, 128009,
            128006, 78191, 128007,
            271, 19045, 29474, 128009,
        ]
        expected_labels = [
            IGNORE_TOKEN_ID,  # bos
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # user header
            IGNORE_TOKEN_ID, 15339, IGNORE_TOKEN_ID,  # user prompt eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # assistant header
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # assistant response eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, 19045, 29474, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
        ]
        # fmt: on

        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        LOG.debug(f"Expected labels: {expected_labels}")
        LOG.debug(f"Actual labels: {labels}")

        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"
        assert (
            labels == expected_labels
        ), f"Labels mismatch: {labels} != {expected_labels}"

    def test_llama3_system_human(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing ShareGPT style datasets with llama-3 system/human prompts")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            train_on_eos="none",
            sequence_len=512,
            roles_to_train=["system", "human"],
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        input_ids = res["input_ids"]
        labels = res["labels"]
        # fmt: off
        expected_input_ids = [
            128000,  # bos
            128006, 9125, 128007,
            271, 2675, 527, 459, 15592, 18328, 13, 128009,
            128006, 882, 128007,  # user header
            271, 9906, 128009,  # user prompt eot
            128006, 78191, 128007,  # assistant header
            271, 13347, 1070, 0, 128009,  # assistant response eot
            128006, 882, 128007,
            271, 4438, 527, 499, 30, 128009,
            128006, 78191, 128007,
            271, 40, 2846, 3815, 1664, 11, 9901, 499, 0, 128009,
        ]
        expected_labels = [
            IGNORE_TOKEN_ID,  # bos
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # system header
            IGNORE_TOKEN_ID, 2675, 527, 459, 15592, 18328, 13, IGNORE_TOKEN_ID,  # system prompt eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # user header
            IGNORE_TOKEN_ID, 9906, IGNORE_TOKEN_ID,  # user prompt eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # assistant header
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,  # assistant response eot
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, 4438, 527, 499, 30, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
        ]
        # fmt: on

        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        LOG.debug(f"Expected labels: {expected_labels}")
        LOG.debug(f"Actual labels: {labels}")

        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"
        assert (
            labels == expected_labels
        ), f"Labels mismatch: {labels} != {expected_labels}"


if __name__ == "__main__":
    unittest.main()
