"""
Tests for handling json tool content
"""

import json

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import (
    load,
)
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="qwen3_instruct_prompt_strategy")
def qwen3_instruct_chat_template_strategy(qwen3_tokenizer):
    strategy = load(
        qwen3_tokenizer,
        DictDefault(
            {
                "train_on_inputs": False,
                "sequence_len": 512,
            }
        ),
        DictDefault(
            {
                "chat_template": "qwen3",
                "message_field_role": "role",
                "message_field_content": "content",
                "message_property_mappings": {
                    "role": "role",
                    "content": "content",
                },
                "roles": {
                    "user": ["user"],
                    "assistant": ["assistant"],
                    "system": ["system"],
                },
                "field_messages": "messages",
            }
        ),
    )
    return strategy


class TestQwen3IdenticalConversationArgs:
    """
    Test Qwen3 tools is identical between JSON and dict
    """

    @pytest.fixture(name="conversation_dict_args_dataset")
    def fixture_conversation_dict_args_dataset(self):
        """
        Provides a dataset with conversation where arguments is a dict.
        """
        user_content = "What is the weather in Boston?"
        function_name = "get_current_weather"
        arguments_dict = {"location": "Boston, MA", "unit": "celsius"}

        data = [
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": function_name,
                                    "arguments": arguments_dict,  # dict格式
                                }
                            }
                        ],
                    },
                ],
            }
        ]
        return Dataset.from_list(data)

    @pytest.fixture(name="conversation_str_args_dataset")
    def fixture_conversation_str_args_dataset(self):
        """
        Provides a dataset with conversation where arguments is a JSON string.
        """
        user_content = "What is the weather in Boston?"
        function_name = "get_current_weather"
        arguments_dict = {"location": "Boston, MA", "unit": "celsius"}
        arguments_str = json.dumps(arguments_dict)

        data = [
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": function_name,
                                    "arguments": arguments_str,  # str格式
                                }
                            }
                        ],
                    },
                ],
            }
        ]
        return Dataset.from_list(data)

    @pytest.fixture(name="conversation_mixed_time_types_dataset")
    def fixture_conversation_mixed_time_types_dataset(self):
        """
        Provides a dataset where 'time' field has different types in different tool calls.
        """
        data = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Get weather information at different times",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "func1",
                                    "arguments": json.dumps(
                                        {"time": "2025-08-01"}
                                    ),  # string type
                                }
                            },
                            {
                                "function": {
                                    "name": "func2",
                                    "arguments": json.dumps(
                                        {"time": 1690876800}
                                    ),  # number type
                                }
                            },
                        ],
                    },
                ],
            }
        ]
        return Dataset.from_list(data)

    def test_dict_and_str_args_produce_identical_output(
        self,
        conversation_dict_args_dataset,
        conversation_str_args_dataset,
        qwen3_instruct_prompt_strategy,
        qwen3_tokenizer,
    ):
        """
        Tests that after tokenization and decoding, the outputs for both
        dict and string `arguments` are exactly the same.
        """
        processed_dict_args = conversation_dict_args_dataset.map(
            qwen3_instruct_prompt_strategy.tokenize_prompt,
            batched=True,
            remove_columns=["messages"],
        )

        processed_str_args = conversation_str_args_dataset.map(
            qwen3_instruct_prompt_strategy.tokenize_prompt,
            batched=True,
            remove_columns=["messages"],
        )

        decoded_prompt_from_dict = qwen3_tokenizer.decode(
            processed_dict_args[0]["input_ids"]
        )

        decoded_prompt_from_str = qwen3_tokenizer.decode(
            processed_str_args[0]["input_ids"]
        )

        assert decoded_prompt_from_dict == decoded_prompt_from_str, (
            f"Dict format output:\n{decoded_prompt_from_dict}\n"
            f"String format output:\n{decoded_prompt_from_str}"
        )

        assert (
            processed_dict_args[0]["input_ids"] == processed_str_args[0]["input_ids"]
        ), "The tokenized input_ids should be identical for dict and str arguments"

    def test_str_args_with_mixed_time_types_no_error(
        self,
        conversation_mixed_time_types_dataset,
        qwen3_instruct_prompt_strategy,
        qwen3_tokenizer,
    ):
        """
        Tests that when 'time' field has different types (string vs number)
        in different tool calls, str format arguments don't cause errors.
        """
        processed = conversation_mixed_time_types_dataset.map(
            qwen3_instruct_prompt_strategy.tokenize_prompt,
            batched=True,
            remove_columns=["messages"],
        )

        assert len(processed) == 1
        assert "input_ids" in processed[0]
        assert len(processed[0]["input_ids"]) > 0

        decoded = qwen3_tokenizer.decode(processed[0]["input_ids"])
        assert "2025-08-01" in decoded, "String time value should be present"
        assert "1690876800" in decoded, "Number time value should be present"
