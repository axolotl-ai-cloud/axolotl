"""Test chat templates for mistral-common wrapper tokenizer"""

import unittest
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from axolotl.utils.mistral import HFMistralTokenizer


# fmt: off
@pytest.mark.parametrize(
    ("tokenizer_str", "assistant_toolcall_ids", "tool_result_ids"),
    (
        ("magistral_tokenizer", (9, 44627, 3684, 33, 19881, 1049, 1050, 1051, 1052, 1053, 32, 19227, 12856, 2811, 1032, 1049, 1054, 1044, 1429, 33319, 2811, 1032, 1050, 1125, 2), (7, 19881, 1049, 1050, 1051, 1052, 1053, 19, 1049, 1044, 1050, 8)),
        ("devstral_tokenizer", (9, 1091, 19227, 2391, 2811, 1429, 44627, 3684, 1897, 1429, 61906, 2811, 16753, 12856, 2811, 1032, 1049, 1054, 1044, 1429, 33319, 2811, 1032, 1050, 4179, 1429, 1327, 2811, 1429, 19881, 1049, 1050, 1051, 1052, 1053, 1034, 27028, 2), (7, 19881, 1049, 1050, 1051, 1052, 1053, 19, 1049, 1044, 1050, 8)),
        ("devstral_1_1_tokenizer", (9, 44627, 3684, 32, 19227, 12856, 2811, 1032, 1049, 1054, 1044, 1429, 33319, 2811, 1032, 1050, 1125, 2,), (7, 1049, 1044, 1050, 8)),
    )
)
# fmt: on
def test_mistral_chat_template(
    tokenizer_str: str,
    assistant_toolcall_ids: tuple[int, ...],
    tool_result_ids: tuple[int, ...],
    request: pytest.FixtureRequest,
):
    """Test chat template with the Magistral/Devstral tokenizer"""

    from axolotl.prompt_strategies.chat_template import MistralPrompter, MistralStrategy

    tokenizer: HFMistralTokenizer = request.getfixturevalue(tokenizer_str)

    # check bos, eos, pad, unk are accessible properties
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 11
    assert tokenizer.unk_token_id == 0

    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.unk_token == "<unk>"

    strategy = MistralStrategy(
        MistralPrompter(
            tokenizer,
            chat_template=None,
            message_property_mappings={"role": "role", "content": "content"},
        ),
        tokenizer=tokenizer,
        train_on_inputs=False,
        train_on_eos="turn",
        sequence_len=512,
        roles_to_train=["assistant"],
    )

    # test chat template masking without system prompt
    res = strategy.tokenize_prompt(
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ]
        }
    )

    assert res["input_ids"] == [
        1,  # bos
        3,  # [INST]
        22177,  # Hello
        1044,  # ,
        2606,  # how
        1584,  # are
        1636,  # you
        1063,  # ?
        4,  # [/INST]
        1073,  # I
        4525,  # 'm
        6965,  # doing
        4824,  # great
        1044,  # ,
        15412,  # thank
        1636,  # you
        1033,  # !
        2,  # </s>
    ]

    assert res["labels"] == [
        -100,  # bos
        -100,  # [INST]
        -100,  # Hello
        -100,  # ,
        -100,  # how
        -100,  # are
        -100,  # you
        -100,  # ?
        -100,  # [/INST]
        1073,  # I
        4525,  # 'm
        6965,  # doing
        4824,  # great
        1044,  # ,
        15412,  # thank
        1636,  # you
        1033,  # !
        2,  # </s>
    ]

    # test chat template masking with system prompt
    res = strategy.tokenize_prompt(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ]
        }
    )

    assert res["input_ids"] == [
        1,  # bos
        17,  # [SYSTEM_PROMPT]
        4568,  # You
        1584,  # are
        1261,  # a
        20351,  # helpful
        27089,  # assistant
        1046,  # .
        18,  # [/SYSTEM_PROMPT]
        3,  # [INST]
        22177,  # Hello
        1044,  # ,
        2606,  # how
        1584,  # are
        1636,  # you
        1063,  # ?
        4,  # [/INST]
        1073,  # I
        4525,  # 'm
        6965,  # doing
        4824,  # great
        1044,  # ,
        15412,  # thank
        1636,  # you
        1033,  # !
        2,  # </s>
    ]

    assert res["labels"] == [
        -100,  # bos
        -100,  # [SYSTEM_PROMPT]
        -100,  # You
        -100,  # are
        -100,  # a
        -100,  # helpful
        -100,  # assistant
        -100,  # .
        -100,  # [/SYSTEM_PROMPT]
        -100,  # [INST]
        -100,  # Hello
        -100,  # ,
        -100,  # how
        -100,  # are
        -100,  # you
        -100,  # ?
        -100,  # [/INST]
        1073,  # I
        4525,  # 'm
        6965,  # doing
        4824,  # great
        1044,  # ,
        15412,  # thank
        1636,  # you
        1033,  # !
        2,  # </s>
    ]

    # test chat template with tools
    res = strategy.tokenize_prompt(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "multiples",
                        "description": "Generates a list of all the multiples of a number that are less than a given limit.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "number": {
                                    "type": "integer",
                                    "description": "The number to find multiples of.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "The upper limit for the multiples.",
                                },
                            },
                            "required": ["number", "limit"],
                        },
                    },
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "Hey, can you give me a breakdown of how to throw an awesome themed party? Like, what themes work best, and how can I set everything up to really wow my guests? I want some ideas on decorations, food, and activities that will make the party unforgettable!",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call12345",
                            "type": "function",
                            "function": {
                                "name": "multiples",
                                "arguments": {
                                    "number": 16,
                                    "limit": 2,
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call12345",
                    "name": "multiples",
                    "content": "1,2",
                },
                {"role": "assistant", "content": "The multiples of 16 is 1 and 2."},
            ],
        }
    )

    # fmt: off
    assert res["input_ids"] == [
        1,  # bos
        5, 1091, 19227, 4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811, 1429, 44627, 3684, 1897, 1429, 14653, 2811, 1429, 10639, 2130, 1261, 2951, 1307, 1747, 1278, 60092, 1307, 1261, 2782, 1455, 1584, 4289, 2224, 1261, 4265, 6139, 39249, 1429, 26204, 2811, 16753, 4994, 2811, 1429, 6371, 1897, 1429, 48649, 2811, 16753, 12856, 2811, 16753, 4994, 2811, 1429, 49039, 1897, 1429, 14653, 2811, 1429, 1784, 2782, 1317, 3081, 60092, 1307, 2613, 4179, 1429, 33319, 2811, 16753, 4994, 2811, 1429, 49039, 1897, 1429, 14653, 2811, 1429, 1784, 9229, 6139, 1394, 1278, 60092, 2613, 47579, 1429, 15760, 2811, 12161, 12856, 1897, 1429, 33319, 4964, 2821, 27028, 6,  # tool prompt
        3, 46634, 1044, 1710, 1636, 5628, 1639, 1261, 44433, 1307, 2606, 1317, 5388, 1420, 54191, 2424, 1286, 8967, 1063, 15621, 1044, 2549, 30305, 2196, 3560, 1044, 1321, 2606, 1710, 1362, 2016, 8605, 2015, 1317, 5524, 118931, 2036, 32951, 1063, 1362, 2933, 2269, 12106, 1408, 101987, 1044, 6939, 1044, 1321, 9216, 1455, 2084, 3180, 1278, 8967, 119141, 1689, 5935, 1033, 4,  # user
        *assistant_toolcall_ids,  # assistant tool calling
        *tool_result_ids,  # tool result
        1784, 60092, 1307, 1032, 1049, 1054, 1395, 1032, 1049, 1321, 1032, 1050, 1046,  # assistant
        2  # eos
    ]

    assert res["labels"] == [
        -100,  # bos
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # tool prompt
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # user prompt
        *assistant_toolcall_ids,  # assistant tool calling
        *([-100] * len(tool_result_ids)),  # tool result
        1784, 60092, 1307, 1032, 1049, 1054, 1395, 1032, 1049, 1321, 1032, 1050, 1046,  # assistant
        2  # eos
    ]
    # fmt: on

    # test chat template with tokenize=False
    res = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great, thank you!"},
        ],
        tokenize=False,
    )

    assert res == "<s>[INST]Hello, how are you?[/INST]I'm doing great, thank you!</s>"

    # test encode
    res = tokenizer.encode("Hello, how are you?", add_special_tokens=True)
    assert res == [
        1,  # bos
        22177,  # Hello
        1044,  # ,
        2606,  # how
        1584,  # are
        1636,  # you
        1063,  # ?
        2,  # eos
    ]

    # test decode no skip special tokens
    decoded_res = tokenizer.decode(res, skip_special_tokens=False)

    assert decoded_res == "<s>Hello, how are you?</s>"

    # test decode skip special tokens
    decoded_res = tokenizer.decode(res, skip_special_tokens=True)
    assert decoded_res == "Hello, how are you?"

    # test encode no special tokens
    res = tokenizer.encode("Hello, how are you?", add_special_tokens=False)
    assert res == [
        22177,  # Hello
        1044,  # ,
        2606,  # how
        1584,  # are
        1636,  # you
        1063,  # ?
    ]

    # test convert ids to tokens
    res = tokenizer.convert_ids_to_tokens(res)
    # spacing are needed as we are converting without decoding
    assert res == ["Hello", ",", " how", " are", " you", "?"]


@pytest.mark.skip(reason="TODO, fix for new HF wrapper call")
def test_magistral_tokenizer_pad_method(magistral_tokenizer: "HFMistralTokenizer"):
    """Test the MistralTokenizer pad method"""
    from axolotl.utils.collators.core import IGNORE_INDEX

    magistral_pad_token_id = 11  # taken from tokenizer.pad_token_id

    # Test padding with input_ids and labels only
    features = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        {"input_ids": [7, 8], "labels": [9, 10]},
    ]

    result = magistral_tokenizer.pad(features, padding=True, return_tensors="pt")

    # Check that input_ids are padded correctly
    assert result["input_ids"].shape == (2, 3)
    assert result["input_ids"].tolist() == [[1, 2, 3], [7, 8, magistral_pad_token_id]]

    # Check that labels are padded correctly
    assert result["labels"].shape == (2, 3)
    assert result["labels"].tolist() == [[4, 5, 6], [9, 10, IGNORE_INDEX]]

    # Check that attention_mask and position_ids are NOT created
    assert "attention_mask" not in result
    assert "position_ids" not in result

    # Test padding with attention_mask
    features_with_attention = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, 8], "labels": [9, 10], "attention_mask": [1, 1]},
    ]

    result = magistral_tokenizer.pad(
        features_with_attention, padding=True, return_tensors="pt"
    )

    # Check that attention_mask is padded correctly
    assert result["attention_mask"].shape == (2, 3)
    assert result["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]

    # Test padding with position_ids
    features_with_position = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6], "position_ids": [0, 1, 2]},
        {"input_ids": [7, 8], "labels": [9, 10], "position_ids": [0, 1]},
    ]

    result = magistral_tokenizer.pad(
        features_with_position, padding=True, return_tensors="pt"
    )

    # Check that position_ids are padded correctly (continuing sequence)
    assert result["position_ids"].shape == (2, 3)
    assert result["position_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]

    # Test padding with all fields
    features_all = [
        {
            "input_ids": [1, 2, 3],
            "labels": [4, 5, 6],
            "attention_mask": [1, 1, 1],
            "position_ids": [0, 1, 2],
        },
        {
            "input_ids": [7, 8],
            "labels": [9, 10],
            "attention_mask": [1, 1],
            "position_ids": [0, 1],
        },
    ]

    result = magistral_tokenizer.pad(features_all, padding=True, return_tensors="pt")

    # All fields should be present and correctly padded
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result
    assert "position_ids" in result

    # Test padding with all sequences same length
    features_same_length = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        {"input_ids": [7, 8, 9], "labels": [10, 11, 12]},
    ]

    result = magistral_tokenizer.pad(
        features_same_length, padding=True, return_tensors="pt"
    )

    # Check match when no padding is needed
    assert result["input_ids"][0].tolist() == features_same_length[0]["input_ids"]
    assert result["labels"][0].tolist() == features_same_length[0]["labels"]

    assert result["input_ids"][1].tolist() == features_same_length[1]["input_ids"]
    assert result["labels"][1].tolist() == features_same_length[1]["labels"]

    # Test padding with max_length parameter
    result = magistral_tokenizer.pad(
        features, padding="max_length", max_length=5, return_tensors="pt"
    )

    # Should pad to max_length
    assert result["input_ids"].shape == (2, 5)
    assert result["labels"].shape == (2, 5)

    # Test numpy return type
    result = magistral_tokenizer.pad(features, padding=True, return_tensors="np")

    # Should return numpy arrays
    import numpy as np

    assert isinstance(result["input_ids"], np.ndarray)
    assert isinstance(result["labels"], np.ndarray)

    # Test unsupported field rejection
    features_unsupported = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6], "unsupported_field": [7, 8, 9]},
    ]

    with pytest.raises(NotImplementedError, match="unsupported_field"):
        magistral_tokenizer.pad(features_unsupported, padding=True, return_tensors="pt")

    # Test token_type_ids rejection
    features_token_type = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6], "token_type_ids": [0, 0, 0]},
    ]

    with pytest.raises(ValueError, match="token_type_ids is not supported"):
        magistral_tokenizer.pad(features_token_type, padding=True, return_tensors="pt")


def test_magistral_tool_calling(magistral_tokenizer: "HFMistralTokenizer"):
    """Test tool calling with the Magistral tokenizer"""
    from axolotl.prompt_strategies.chat_template import MistralPrompter, MistralStrategy

    strategy = MistralStrategy(
        MistralPrompter(
            magistral_tokenizer,
            chat_template=None,
            message_property_mappings={"role": "role", "content": "content"},
        ),
        tokenizer=magistral_tokenizer,
        train_on_inputs=False,
        train_on_eos="turn",
        sequence_len=512,
        roles_to_train=["assistant"],
    )

    # Test basic tool calling with single function
    basic_tool_calling = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call12345",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "location": "San Francisco, CA",
                            },
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call12345",
                "name": "get_weather",
                "content": "Sunny, 72°F",
            },
            {
                "role": "assistant",
                "content": "The weather in San Francisco is sunny and 72°F.",
            },
        ],
    }

    res = strategy.tokenize_prompt(basic_tool_calling)

    # Basic validation
    assert "input_ids" in res
    assert "labels" in res
    assert len(res["input_ids"]) > 0
    assert len(res["labels"]) == len(res["input_ids"])

    # Decode and verify structure
    decoded = magistral_tokenizer.decode(res["input_ids"])
    assert (
        '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}}, "required": ["location"]}}}][/AVAILABLE_TOOLS]'
        in decoded
    )
    assert (
        '[TOOL_CALLS]get_weather[CALL_ID]call12345[ARGS]{"location": "San Francisco, CA"}</s>'
        in decoded
    )
    assert "[TOOL_RESULTS]call12345[TOOL_CONTENT]Sunny, 72°F[/TOOL_RESULTS]" in decoded
    assert "The weather in San Francisco is sunny and 72°F.</s>" in decoded

    # Test multiple tool calls in sequence
    multi_tool_calling = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply_numbers",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "First number"},
                            "y": {"type": "number", "description": "Second number"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": "Add 5 and 3, then multiply the result by 2",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call12345",
                        "type": "function",
                        "function": {
                            "name": "add_numbers",
                            "arguments": {"a": 5, "b": 3},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call12345",
                "name": "add_numbers",
                "content": "8",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call23456",
                        "type": "function",
                        "function": {
                            "name": "multiply_numbers",
                            "arguments": {"x": 8, "y": 2},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call23456",
                "name": "multiply_numbers",
                "content": "16",
            },
            {
                "role": "assistant",
                "content": "The result is 16. I first added 5 and 3 to get 8, then multiplied 8 by 2 to get 16.",
            },
        ],
    }

    res = strategy.tokenize_prompt(multi_tool_calling)

    # Validation
    assert len(res["input_ids"]) > 0
    assert len(res["labels"]) == len(res["input_ids"])

    decoded = magistral_tokenizer.decode(res["input_ids"])
    assert (
        '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "add_numbers", "description": "Add two numbers together", "parameters": {"type": "object", "properties": {"a": {"type": "number", "description": "First number"}, "b": {"type": "number", "description": "Second number"}}, "required": ["a", "b"]}}}, {"type": "function", "function": {"name": "multiply_numbers", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "First number"}, "y": {"type": "number", "description": "Second number"}}, "required": ["x", "y"]}}}][/AVAILABLE_TOOLS]'
        in decoded
    )
    assert (
        '[TOOL_CALLS]add_numbers[CALL_ID]call12345[ARGS]{"a": 5, "b": 3}</s>' in decoded
    )
    assert "[TOOL_RESULTS]call12345[TOOL_CONTENT]8[/TOOL_RESULTS]" in decoded
    assert (
        '[TOOL_CALLS]multiply_numbers[CALL_ID]call23456[ARGS]{"x": 8, "y": 2}</s>'
        in decoded
    )
    assert "[TOOL_RESULTS]call23456[TOOL_CONTENT]16[/TOOL_RESULTS]" in decoded
    assert (
        "The result is 16. I first added 5 and 3 to get 8, then multiplied 8 by 2 to get 16.</s>"
        in decoded
    )

    # Test tool calling with system message
    system_tool_calling = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search for information in database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ],
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a database.",
            },
            {
                "role": "user",
                "content": "Find information about Python programming",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "search123",
                        "type": "function",
                        "function": {
                            "name": "search_database",
                            "arguments": {"query": "Python programming"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "search123",
                "name": "search_database",
                "content": "Python is a high-level programming language known for its simplicity.",
            },
            {
                "role": "assistant",
                "content": "Based on the database search, Python is a high-level programming language known for its simplicity and readability.",
            },
        ],
    }

    res = strategy.tokenize_prompt(system_tool_calling)

    # Validation
    assert len(res["input_ids"]) > 0
    assert len(res["labels"]) == len(res["input_ids"])

    decoded = magistral_tokenizer.decode(res["input_ids"])

    assert (
        '<s>[SYSTEM_PROMPT]You are a helpful assistant with access to a database.[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"type": "function", "function": {"name": "search_database", "description": "Search for information in database", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}}][/AVAILABLE_TOOLS]'
        in decoded
    )

    # Test error handling - missing tool response
    incomplete_tool_calling = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": "What time is it?",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "time12345",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": {},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The current time is 12:00 PM.",
            },
        ],
    }

    from mistral_common.exceptions import InvalidMessageStructureException

    try:
        strategy.tokenize_prompt(incomplete_tool_calling)
    except InvalidMessageStructureException as e:
        assert "Not the same number of function calls and responses" in str(e)


@pytest.mark.skip(reason="TODO, fix for new HF wrapper call")
def test_magistral_tokenizer_call_method(
    magistral_tokenizer: "HFMistralTokenizer", llama3_tokenizer: "PreTrainedTokenizer"
):
    """Test the __call__ method behavior matches HuggingFace standards"""
    from copy import deepcopy

    import numpy as np
    import torch

    hf_tokenizer = deepcopy(llama3_tokenizer)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token

    test_text = "Hello, how are you?"
    batch_texts = ["Hello world", "How are you?"]

    # Test single string with return_tensors=None
    hf_result: dict[str, list[int]] = hf_tokenizer(test_text, return_tensors=None)
    mistral_result: dict[str, list[int]] = magistral_tokenizer(
        test_text, return_tensors=None
    )

    assert isinstance(mistral_result, dict)
    assert set(mistral_result.keys()) == {"input_ids", "attention_mask"}
    assert isinstance(mistral_result["input_ids"], type(hf_result["input_ids"]))  # list
    assert isinstance(
        mistral_result["attention_mask"], type(hf_result["attention_mask"])
    )
    assert len(mistral_result["input_ids"]) == len(mistral_result["attention_mask"])
    assert np.all(mistral_result["attention_mask"])
    assert len(np.array(mistral_result["input_ids"]).shape) == 1  # 1D array

    # Test single string with return_tensors='pt'
    hf_result_pt: dict[str, torch.Tensor] = hf_tokenizer(test_text, return_tensors="pt")
    mistral_result_pt: dict[str, torch.Tensor] = magistral_tokenizer(
        test_text, return_tensors="pt"
    )

    # Check structure and types
    assert isinstance(mistral_result_pt["input_ids"], torch.Tensor)
    assert isinstance(mistral_result_pt["attention_mask"], torch.Tensor)

    # Check shapes match (don't compare token dimension)
    assert len(hf_result_pt["input_ids"].shape) == len(
        mistral_result_pt["input_ids"].shape
    )
    assert hf_result_pt["input_ids"].shape[0] == mistral_result_pt["input_ids"].shape[0]
    assert (
        mistral_result_pt["attention_mask"].shape
        == mistral_result_pt["input_ids"].shape
    )
    assert torch.all(mistral_result_pt["attention_mask"] == 1)

    # Test batch input with padding
    hf_batch: dict[str, torch.Tensor] = hf_tokenizer(
        batch_texts, return_tensors="pt", padding=True
    )
    mistral_batch: dict[str, torch.Tensor] = magistral_tokenizer(
        batch_texts, return_tensors="pt", padding=True
    )

    # Check batch behavior
    assert len(hf_batch["input_ids"].shape) == len(mistral_batch["input_ids"].shape)
    assert hf_batch["input_ids"].shape[0] == mistral_batch["input_ids"].shape[0]
    assert mistral_batch["attention_mask"].shape == mistral_batch["input_ids"].shape
    assert torch.any(
        mistral_batch["attention_mask"][0] == 0
    )  # padding in shorter sequence
    assert torch.all(
        mistral_batch["attention_mask"][1] == 1
    )  # no padding in longer sequence

    # Test numpy tensors
    mistral_result_np: dict[str, np.ndarray] = magistral_tokenizer(
        test_text, return_tensors="np"
    )
    assert isinstance(mistral_result_np["input_ids"], np.ndarray)
    assert isinstance(mistral_result_np["attention_mask"], np.ndarray)

    # Test consistency with encode()
    encoded: list[int] = magistral_tokenizer.encode(test_text, add_special_tokens=True)
    called: dict[str, torch.Tensor] = magistral_tokenizer(
        test_text, return_tensors="pt"
    )
    assert encoded == called["input_ids"][0].tolist()

    # Test Error handling
    with pytest.raises(ValueError, match="Unsupported kwargs"):
        magistral_tokenizer(test_text, unsupported_param=True)

    with pytest.raises(
        ValueError, match="return_tensors='pt' or 'np' requires padding or truncation"
    ):
        magistral_tokenizer(batch_texts, return_tensors="pt")


if __name__ == "__main__":
    unittest.main()
