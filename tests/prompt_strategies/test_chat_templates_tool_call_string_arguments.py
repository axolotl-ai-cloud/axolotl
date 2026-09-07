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
                                    "arguments": arguments_dict,  # dict
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
                                    "arguments": arguments_str,  # str
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


class TestQwen3IdenticalToolsParameters:
    """
    Test Qwen3 tools parameters handling is identical between JSON string and dict
    """

    @pytest.fixture(name="tools_dict_params_dataset")
    def fixture_tools_dict_params_dataset(self):
        """
        Provides a dataset with tools where parameters is a dict.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        data = [
            {
                "tools": tools,
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"location": "Boston, MA"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "get_weather",
                        "content": "72°F and sunny",
                    },
                ],
            }
        ]
        return Dataset.from_list(data)

    @pytest.fixture(name="tools_str_params_dataset")
    def fixture_tools_str_params_dataset(self):
        """
        Provides a dataset with tools where parameters is a JSON string.
        """
        parameters_dict = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": json.dumps(parameters_dict),
                },
            }
        ]

        data = [
            {
                "tools": tools,
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"location": "Boston, MA"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "get_weather",
                        "content": "72°F and sunny",
                    },
                ],
            }
        ]
        return Dataset.from_list(data)

    @pytest.fixture(name="tools_mixed_type_params_dataset")
    def fixture_tools_mixed_type_params_dataset(self):
        """
        Provides a dataset where different tools have the same parameter name with different types.
        This tests that JSON string format prevents casting issues.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_with_string_arg",
                    "description": "Tool expecting string argument",
                    "parameters": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "arg1": {
                                    "type": "string",
                                    "description": "A string parameter",
                                }
                            },
                            "required": ["arg1"],
                        }
                    ),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_with_number_arg",
                    "description": "Tool expecting number argument",
                    "parameters": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "arg1": {
                                    "type": "number",
                                    "description": "A numeric parameter",
                                }
                            },
                            "required": ["arg1"],
                        }
                    ),
                },
            },
        ]

        data = [
            {
                "tools": tools,
                "messages": [
                    {"role": "user", "content": "Use both tools"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "tool_with_string_arg",
                                    "arguments": json.dumps({"arg1": "hello"}),
                                },
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "tool_with_number_arg",
                                    "arguments": json.dumps({"arg1": 42}),
                                },
                            },
                        ],
                    },
                ],
            }
        ]
        return Dataset.from_list(data)

    def test_dict_and_str_params_produce_equivalent_output(
        self,
        tools_dict_params_dataset,
        tools_str_params_dataset,
        qwen3_instruct_prompt_strategy,
        qwen3_tokenizer,
    ):
        """
        Tests that after tokenization and decoding, the outputs for both
        dict and string `parameters` in tools are semantically equivalent.
        """
        import re

        processed_dict_params = tools_dict_params_dataset.map(
            qwen3_instruct_prompt_strategy.tokenize_prompt,
            batched=True,
            remove_columns=["messages", "tools"],
        )

        processed_str_params = tools_str_params_dataset.map(
            qwen3_instruct_prompt_strategy.tokenize_prompt,
            batched=True,
            remove_columns=["messages", "tools"],
        )

        decoded_dict = qwen3_tokenizer.decode(processed_dict_params[0]["input_ids"])
        decoded_str = qwen3_tokenizer.decode(processed_str_params[0]["input_ids"])

        # Extract the tool JSON from both outputs
        tools_pattern = r"<tools>\n(.*?)\n</tools>"

        dict_tools_match = re.search(tools_pattern, decoded_dict, re.DOTALL)
        str_tools_match = re.search(tools_pattern, decoded_str, re.DOTALL)

        assert dict_tools_match and str_tools_match, (
            "Could not find tools section in output"
        )

        # Parse the JSON and compare as objects (order-independent)
        dict_tools_json = json.loads(dict_tools_match.group(1))
        str_tools_json = json.loads(str_tools_match.group(1))

        # Deep comparison of the tool definitions
        assert dict_tools_json == str_tools_json, (
            f"Tool definitions are not equivalent:\n"
            f"Dict format: {json.dumps(dict_tools_json, indent=2)}\n"
            f"String format: {json.dumps(str_tools_json, indent=2)}"
        )

        # Verify the rest of the structure is the same (excluding the tools JSON part)
        # The tools JSON can have different order, so we remove it here.
        dict_normalized = re.sub(
            r"<tools>.*?</tools>",
            "<tools>TOOLS_PLACEHOLDER</tools>",
            decoded_dict,
            flags=re.DOTALL,
        )
        str_normalized = re.sub(
            r"<tools>.*?</tools>",
            "<tools>TOOLS_PLACEHOLDER</tools>",
            decoded_str,
            flags=re.DOTALL,
        )

        assert dict_normalized == str_normalized, (
            "The overall structure differs between dict and string parameter formats"
        )

    def test_str_params_with_mixed_types_no_error(
        self,
        tools_mixed_type_params_dataset,
        qwen3_instruct_prompt_strategy,
        qwen3_tokenizer,
    ):
        """
        Tests that when different tools have the same parameter name with different types,
        JSON string format for parameters doesn't cause casting errors.
        """
        processed = tools_mixed_type_params_dataset.map(
            qwen3_instruct_prompt_strategy.tokenize_prompt,
            batched=True,
            remove_columns=["messages", "tools"],
        )

        assert len(processed) == 1
        assert "input_ids" in processed[0]
        assert len(processed[0]["input_ids"]) > 0

        decoded = qwen3_tokenizer.decode(processed[0]["input_ids"])

        # Check that both tools are present
        assert "tool_with_string_arg" in decoded
        assert "tool_with_number_arg" in decoded

        # Check that both argument values are present
        assert "hello" in decoded
        assert "42" in decoded
