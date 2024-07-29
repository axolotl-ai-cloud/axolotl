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


class TestChatTemplateConfigurations:
    """
    Test class for various configurations of ChatTemplateStrategy.
    """

    @staticmethod
    def find_sublist(full_list, sub_list):
        token_count = len(sub_list)
        for index in range(len(full_list) - token_count + 1):
            if full_list[index : index + token_count] == sub_list:
                return index
        return -1

    def test_train_on_inputs_true(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with train_on_inputs=True")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=True,
            sequence_len=512,
            roles_to_train=["assistant"],
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Verify that assistant responses are labeled
        assistant_responses = ["Hi there!", "I'm doing well, thank you!"]
        for response in assistant_responses:
            response_ids = llama3_tokenizer.encode(response, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, response_ids)
            LOG.debug(
                f"Assistant response '{response}' expected IDs: {response_ids}, found at: {start_idx}"
            )
            assert start_idx != -1, f"Could not find '{response}' in input_ids"
            assert all(
                label != IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(response_ids)]
            ), f"Expected labels for assistant response '{response}' to be set, but got {labels[start_idx:start_idx+len(response_ids)]}"

        # Check the behavior of human inputs
        human_inputs = ["Hello", "How are you?"]
        for input_text in human_inputs:
            input_ids = llama3_tokenizer.encode(input_text, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, input_ids)
            labeled = all(
                label != IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(input_ids)]
            )
            LOG.debug(
                f"Human input '{input_text}' is {'labeled' if labeled else 'not labeled'}, expected IDs: {input_ids}, found at: {start_idx}"
            )

        LOG.debug("Full labels: %s", labels)
        LOG.debug("Full input_ids: %s", input_ids)

    def test_train_on_inputs_false(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with train_on_inputs=False")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Verify that only assistant responses are labeled
        assistant_responses = ["Hi there!", "I'm doing well, thank you!"]
        for response in assistant_responses:
            response_ids = llama3_tokenizer.encode(response, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, response_ids)
            LOG.debug(
                f"Assistant response '{response}' expected IDs: {response_ids}, found at: {start_idx}"
            )
            assert start_idx != -1, f"Could not find '{response}' in input_ids"
            assert all(
                label != IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(response_ids)]
            ), f"Expected labels for assistant response '{response}' to be set, but got {labels[start_idx:start_idx+len(response_ids)]}"

        # Verify that human inputs are not labeled
        human_inputs = ["Hello", "How are you?"]
        for input_text in human_inputs:
            input_ids = llama3_tokenizer.encode(input_text, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, input_ids)
            LOG.debug(
                f"Human input '{input_text}' expected IDs: {input_ids}, found at: {start_idx}"
            )
            assert start_idx != -1, f"Could not find '{input_text}' in input_ids"
            assert all(
                label == IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(input_ids)]
            ), f"Expected labels for human input '{input_text}' to be IGNORE_TOKEN_ID, but got {labels[start_idx:start_idx+len(input_ids)]}"

    def test_roles_to_train_assistant_only(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing roles_to_train with assistant only")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Verify that only assistant responses are labeled
        assistant_responses = ["Hi there!", "I'm doing well, thank you!"]
        for response in assistant_responses:
            response_ids = llama3_tokenizer.encode(response, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, response_ids)
            LOG.debug(
                f"Assistant response '{response}' expected IDs: {response_ids}, found at: {start_idx}"
            )
            assert all(
                label != IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(response_ids)]
            ), f"Expected labels for assistant response '{response}' to be set, but got {labels[start_idx:start_idx+len(response_ids)]}"

    def test_roles_to_train_all(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing roles_to_train with all roles")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=True,
            sequence_len=512,
            roles_to_train=["human", "assistant"],
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Verify that all responses are labeled (except for special tokens)
        all_responses = [
            "Hello",
            "Hi there!",
            "How are you?",
            "I'm doing well, thank you!",
        ]
        for response in all_responses:
            response_ids = llama3_tokenizer.encode(response, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, response_ids)
            LOG.debug(
                f"Response '{response}' expected IDs: {response_ids}, found at: {start_idx}"
            )
            assert all(
                label != IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(response_ids)]
            ), f"Expected labels for response '{response}' to be set, but got {labels[start_idx:start_idx+len(response_ids)]}"

    def test_empty_roles_to_train(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with empty roles_to_train")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=[],
            train_on_eos="none",  # Add this line
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]

        # Verify that no labels are set when roles_to_train is empty
        LOG.debug("Full labels: %s", labels)
        assert all(
            label == IGNORE_TOKEN_ID for label in labels
        ), "Expected all labels to be IGNORE_TOKEN_ID when roles_to_train is empty"

    def test_train_on_eos_all(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with train_on_eos='all'")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="all",
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = llama3_tokenizer.eos_token_id
        eos_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
        ]

        assert len(eos_indices) > 0, "Expected at least one EOS token in the input"
        for eos_idx in eos_indices:
            assert (
                labels[eos_idx] != IGNORE_TOKEN_ID
            ), f"Expected EOS token at index {eos_idx} to be labeled"

    def test_train_on_eos_turn(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with train_on_eos='turn'")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="turn",
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = llama3_tokenizer.eos_token_id
        assistant_responses = ["Hi there!", "I'm doing well, thank you!"]

        for response in assistant_responses:
            response_ids = llama3_tokenizer.encode(response, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, response_ids)
            assert start_idx != -1, f"Could not find '{response}' in input_ids"

            eos_idx = start_idx + len(response_ids)
            while eos_idx < len(input_ids) and input_ids[eos_idx] != eos_token_id:
                eos_idx += 1

            assert eos_idx < len(
                input_ids
            ), f"Could not find EOS token after '{response}'"
            assert (
                labels[eos_idx] != IGNORE_TOKEN_ID
            ), f"Expected EOS token after assistant response '{response}' to be labeled"

        # Check that EOS tokens after human inputs are not labeled
        human_inputs = ["Hello", "How are you?"]
        for input_text in human_inputs:
            input_ids = llama3_tokenizer.encode(input_text, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, input_ids)
            assert start_idx != -1, f"Could not find '{input_text}' in input_ids"

            eos_idx = start_idx + len(input_ids)
            while eos_idx < len(input_ids) and input_ids[eos_idx] != eos_token_id:
                eos_idx += 1

            assert (
                labels[eos_idx] == IGNORE_TOKEN_ID
            ), f"Expected EOS token after human input '{input_text}' to not be labeled"

    def test_train_on_eos_last(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with train_on_eos='last'")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="last",
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = llama3_tokenizer.eos_token_id
        eos_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
        ]

        assert len(eos_indices) > 0, "Expected at least one EOS token in the input"
        last_eos_idx = eos_indices[-1]

        # Check that only the last EOS token is labeled
        for idx in eos_indices[:-1]:
            assert (
                labels[idx] == IGNORE_TOKEN_ID
            ), f"Expected EOS token at index {idx} to not be labeled"
        assert (
            labels[last_eos_idx] != IGNORE_TOKEN_ID
        ), f"Expected last EOS token at index {last_eos_idx} to be labeled"

    def test_train_on_eos_none(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with train_on_eos='none'")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(llama3_tokenizer, chat_templates("llama3")),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
            train_on_eos="none",
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        eos_token_id = llama3_tokenizer.eos_token_id
        eos_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == eos_token_id
        ]

        assert len(eos_indices) > 0, "Expected at least one EOS token in the input"
        for eos_idx in eos_indices:
            assert (
                labels[eos_idx] == IGNORE_TOKEN_ID
            ), f"Expected EOS token at index {eos_idx} to not be labeled"

    def test_drop_system_message(self, llama3_tokenizer, basic_dataset):
        LOG.info("Testing with drop_system_message=True")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                llama3_tokenizer, chat_templates("llama3"), drop_system_message=True
            ),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["assistant"],
        )
        res = strategy.tokenize_prompt(basic_dataset[0])
        input_ids = res["input_ids"]

        # Check if system message is not present in input_ids
        system_message = "You are an AI assistant."
        system_ids = llama3_tokenizer.encode(system_message, add_special_tokens=False)
        assert (
            self.find_sublist(input_ids, system_ids) == -1
        ), "Expected system message to be dropped"

    def test_custom_roles(self, llama3_tokenizer):
        LOG.info("Testing with custom roles mapping")
        custom_roles = {
            "user": ["human", "user"],
            "assistant": ["ai", "assistant"],
            "system": ["context"],
        }
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                llama3_tokenizer, chat_templates("llama3"), roles=custom_roles
            ),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=["ai"],
        )

        # Create a new dataset with modified role names
        modified_conversations = [
            {"from": "context", "value": "You are an AI assistant."},
            {"from": "human", "value": "Hello"},
            {"from": "ai", "value": "Hi there!"},
            {"from": "human", "value": "How are you?"},
            {"from": "ai", "value": "I'm doing well, thank you!"},
        ]

        modified_dataset = Dataset.from_dict(
            {"conversations": [modified_conversations]}
        )

        res = strategy.tokenize_prompt(modified_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Check if AI responses are labeled correctly
        ai_responses = ["Hi there!", "I'm doing well, thank you!"]
        for response in ai_responses:
            response_ids = llama3_tokenizer.encode(response, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, response_ids)
            assert start_idx != -1, f"Could not find response '{response}' in input_ids"
            assert all(
                label != IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(response_ids)]
            ), f"Expected labels for AI response '{response}' to be set"

        # Check if human messages are not labeled
        human_messages = ["Hello", "How are you?"]
        for message in human_messages:
            message_ids = llama3_tokenizer.encode(message, add_special_tokens=False)
            start_idx = self.find_sublist(input_ids, message_ids)
            assert start_idx != -1, f"Could not find message '{message}' in input_ids"
            assert all(
                label == IGNORE_TOKEN_ID
                for label in labels[start_idx : start_idx + len(message_ids)]
            ), f"Expected labels for human message '{message}' to be IGNORE_TOKEN_ID"

    def test_message_field_training(self, llama3_tokenizer):
        LOG.info("Testing with message_field_training")
        strategy = ChatTemplateStrategy(
            ChatTemplatePrompter(
                llama3_tokenizer,
                chat_templates("llama3"),
                message_field_training="train",
                message_field_training_detail="train_detail",
            ),
            tokenizer=llama3_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            roles_to_train=[],
        )

        # Create a new dataset with the train and train_detail fields
        modified_conversation = [
            {"from": "system", "value": "You are an AI assistant.", "train": False},
            {"from": "human", "value": "Hello", "train": False},
            {"from": "assistant", "value": "Hello", "train": True},
            {"from": "human", "value": "How are you?", "train": True},
            {
                "from": "assistant",
                "value": "I'm doing very well, thank you!",
                "train_detail": [
                    {"begin_offset": 0, "end_offset": 8, "train": False},
                    {"begin_offset": 9, "end_offset": 18, "train": True},
                    {"begin_offset": 19, "end_offset": 30, "train": False},
                ],
            },
            {
                "from": "human",
                "value": "I'm doing very well, thank you!",
                "train": False,
            },
            {"from": "assistant", "value": "Hi there!", "train": True},
        ]

        modified_dataset = Dataset.from_dict({"conversations": [modified_conversation]})

        res = strategy.tokenize_prompt(modified_dataset[0])
        labels = res["labels"]
        input_ids = res["input_ids"]

        # Function to find all occurrences of a sublist
        def find_all_sublists(full_list, sub_list):
            indices = []
            for index in range(len(full_list) - len(sub_list) + 1):
                if full_list[index : index + len(sub_list)] == sub_list:
                    indices.append(index)
            return indices

        # Keep track of which occurrences we've processed
        processed_occurrences = {}
        # Check if messages are labeled correctly based on train or train_detail
        for i, turn in enumerate(modified_conversation):
            turn_tokens = llama3_tokenizer.encode(
                turn["value"], add_special_tokens=False
            )
            occurrences = find_all_sublists(input_ids, turn_tokens)
            turn_key = turn["value"]
            if turn_key not in processed_occurrences:
                processed_occurrences[turn_key] = 0
            current_occurrence = processed_occurrences[turn_key]

            if current_occurrence >= len(occurrences):
                assert (
                    False
                ), f"Not enough occurrences found for message: {turn['value']}"

            start_idx = occurrences[current_occurrence]
            processed_occurrences[turn_key] += 1
            end_idx = start_idx + len(turn_tokens)

            LOG.debug(
                f"Processing turn {i}: role={turn['from']}, content='{turn['value']}', start_idx={start_idx}, end_idx={end_idx}"
            )

            if "train_detail" in turn:
                # Get token offsets
                tokenized_output = llama3_tokenizer(
                    turn["value"], return_offsets_mapping=True, add_special_tokens=False
                )
                token_offsets = tokenized_output["offset_mapping"]

                # Adjust token offsets as done in the implementation
                for i in range(len(token_offsets) - 1):
                    token_offsets[i] = (
                        token_offsets[i][0],
                        token_offsets[i + 1][0] - 1,
                    )
                token_offsets[-1] = (token_offsets[-1][0], len(turn["value"]) - 1)

                # Adjust train_details
                adjusted_train_details = strategy.prompter.adjust_train_details(
                    turn["train_detail"], token_offsets
                )

                LOG.debug(f"Original train_details: {turn['train_detail']}")
                LOG.debug(f"Adjusted train_details: {adjusted_train_details}")

                # Handle train_detail
                token_offsets = strategy.prompter.get_offsets_for_train_detail(
                    text=turn["value"],
                    train_details=adjusted_train_details,
                    mask_untrainable=False,
                )
                token_offsets_masked = strategy.prompter.get_offsets_for_train_detail(
                    text=turn["value"],
                    train_details=adjusted_train_details,
                    mask_untrainable=True,
                )
                LOG.debug(f"Token offsets: {token_offsets_masked}")

                expected_labels = [IGNORE_TOKEN_ID] * len(turn_tokens)
                for i, offset in enumerate(token_offsets_masked):
                    if offset != IGNORE_TOKEN_ID:
                        expected_labels[i] = turn_tokens[i]
                actual_labels = labels[
                    start_idx : start_idx + len(token_offsets_masked)
                ]
                assert (
                    actual_labels == expected_labels
                ), f"Labels mismatch for turn: {turn['value']}\nExpected: {expected_labels}\nActual: {actual_labels}"

                for detail in adjusted_train_details:
                    # Find the token indices that correspond to the character offsets
                    detail_start = start_idx + next(
                        i
                        for i, offset in enumerate(token_offsets)
                        if offset >= detail["begin_offset"]
                    )
                    detail_end = start_idx + next(
                        (
                            i
                            for i, offset in enumerate(token_offsets)
                            if offset > detail["end_offset"]
                        ),
                        len(token_offsets),
                    )

                    detail_text = turn["value"][
                        detail["begin_offset"] : detail["end_offset"] + 1
                    ]
                    detail_labels = labels[detail_start:detail_end]
                    detail_input_ids = input_ids[detail_start:detail_end]

                    LOG.debug(
                        f"Detail: '{detail_text}', Start: {detail_start}, End: {detail_end}"
                    )
                    LOG.debug(f"Detail input_ids: {detail_input_ids}")
                    LOG.debug(f"Detail labels: {detail_labels}")
                    LOG.debug(
                        f"Decoded detail: {llama3_tokenizer.decode(detail_input_ids)}"
                    )
                    LOG.debug(
                        f"Token offsets for this detail: {token_offsets[detail_start-start_idx:detail_end-start_idx]}"
                    )

                    if detail["train"]:
                        assert all(
                            label != IGNORE_TOKEN_ID for label in detail_labels
                        ), (
                            f"Expected labels for trainable detail '{detail_text}' to be set, but some were IGNORE_TOKEN_ID. "
                            f"Labels({detail_start}:{detail_end}): {detail_labels}, "
                            f"InputIDs: {detail_input_ids}, "
                            f"Decoded: '{llama3_tokenizer.decode(detail_input_ids)}'"
                        )
                    else:
                        assert all(
                            label == IGNORE_TOKEN_ID for label in detail_labels
                        ), (
                            f"Expected all labels for non-trainable detail '{detail_text}' to be IGNORE_TOKEN_ID, but some were not. "
                            f"Labels({detail_start}:{detail_end}): {detail_labels}, "
                            f"InputIDs: {detail_input_ids}, "
                            f"Decoded: '{llama3_tokenizer.decode(detail_input_ids)}'"
                        )
            else:
                should_train = turn.get("train", False)
                turn_labels = labels[start_idx:end_idx]

                LOG.debug(f"Should train: {should_train}")
                LOG.debug(f"Turn indices: start={start_idx}, end={end_idx}")
                LOG.debug(f"Turn labels: {turn_labels}")
                LOG.debug(f"Turn input IDs: {input_ids[start_idx:end_idx]}")
                LOG.debug(
                    f"Decoded turn: {llama3_tokenizer.decode(input_ids[start_idx:end_idx])}"
                )

                if should_train:
                    assert all(label != IGNORE_TOKEN_ID for label in turn_labels), (
                        f"Expected all labels for '{turn['value']}' to be set\n"
                        f"Labels({start_idx}:{end_idx}): {turn_labels}, "
                        f"InputIDs: {input_ids[start_idx:end_idx]}, "
                        f"Decoded: '{llama3_tokenizer.decode(input_ids[start_idx:end_idx])}'"
                    )
                else:
                    assert all(label == IGNORE_TOKEN_ID for label in turn_labels), (
                        f"Expected all labels for '{turn['value']}' to be IGNORE_TOKEN_ID\n"
                        f"Labels({start_idx}:{end_idx}): {turn_labels}, "
                        f"InputIDs: {input_ids[start_idx:end_idx]}, "
                        f"Decoded: '{llama3_tokenizer.decode(input_ids[start_idx:end_idx])}'"
                    )

                LOG.debug(
                    f"Processed turn: {turn['from']}, content: '{turn['value']}', "
                    f"start_idx: {start_idx}, end_idx: {end_idx}, "
                    f"labels: {labels[start_idx:end_idx]}"
                )

        LOG.debug(f"Final labels: {labels}")
        LOG.debug(f"Final input_ids: {input_ids}")


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
            roles_to_train=["assistant"],
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
