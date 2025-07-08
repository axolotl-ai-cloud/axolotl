"""Test chat templates for mistral-common wrapper tokenizer"""

import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axolotl.utils.mistral_tokenizer import HFMistralTokenizer


def test_magistral_chat_template(magistral_tokenizer: "HFMistralTokenizer"):
    # pylint: disable=duplicate-code
    from axolotl.prompt_strategies.chat_template import MistralPrompter, MistralStrategy

    # check bos, eos, pad, unk are accessible properties
    assert magistral_tokenizer.bos_token_id == 1
    assert magistral_tokenizer.eos_token_id == 2
    assert magistral_tokenizer.pad_token_id == 11
    assert magistral_tokenizer.unk_token_id == 0

    assert magistral_tokenizer.pad_token == "<pad>"
    assert magistral_tokenizer.eos_token == "</s>"
    assert magistral_tokenizer.bos_token == "<s>"
    assert magistral_tokenizer.unk_token == "<unk>"

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
        9, 44627, 3684, 33, 19881, 1049, 1050, 1051, 1052, 1053, 32, 19227, 12856, 2811, 1032, 1049, 1054, 1044, 1429, 33319, 2811, 1032, 1050, 1125, 2,  # assistant tool calling
        7, 19881, 1049, 1050, 1051, 1052, 1053, 19, 1049, 1044, 1050, 8,  # tool result
        1784, 60092, 1307, 1032, 1049, 1054, 1395, 1032, 1049, 1321, 1032, 1050, 1046,  # assistant
        2  # eos
    ]

    assert res["labels"] == [
        -100,  # bos
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # tool prompt
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # user prompt
        9, 44627, 3684, 33, 19881, 1049, 1050, 1051, 1052, 1053, 32, 19227, 12856, 2811, 1032, 1049, 1054, 1044, 1429, 33319, 2811, 1032, 1050, 1125, 2,  # assistant tool calling
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # tool result
        1784, 60092, 1307, 1032, 1049, 1054, 1395, 1032, 1049, 1321, 1032, 1050, 1046,  # assistant
        2  # eos
    ]
    # fmt: on

    # test chat template with tokenize=False
    res = magistral_tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great, thank you!"},
        ],
        tokenize=False,
    )

    assert res == "<s>[INST]Hello, how are you?[/INST]I'm doing great, thank you!</s>"

    # test encode
    res = magistral_tokenizer.encode("Hello, how are you?", add_special_tokens=True)
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
    decoded_res = magistral_tokenizer.decode(res, skip_special_tokens=False)

    assert decoded_res == "<s>Hello, how are you?</s>"

    # test decode skip special tokens
    decoded_res = magistral_tokenizer.decode(res, skip_special_tokens=True)
    assert decoded_res == "Hello, how are you?"

    # test encode no special tokens
    res = magistral_tokenizer.encode("Hello, how are you?", add_special_tokens=False)
    assert res == [
        22177,  # Hello
        1044,  # ,
        2606,  # how
        1584,  # are
        1636,  # you
        1063,  # ?
    ]

    # test convert ids to tokens
    res = magistral_tokenizer.convert_ids_to_tokens(res)
    # spacing are needed as we are converting without decoding
    assert res == ["Hello", ",", " how", " are", " you", "?"]


def test_mistral_tokenizer_pad_method(magistral_tokenizer: "HFMistralTokenizer"):
    """Test the pad method with various field combinations."""
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

    try:
        magistral_tokenizer.pad(features_unsupported, padding=True, return_tensors="pt")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "unsupported_field" in str(e)

    # Test token_type_ids rejection
    features_token_type = [
        {"input_ids": [1, 2, 3], "labels": [4, 5, 6], "token_type_ids": [0, 0, 0]},
    ]

    try:
        magistral_tokenizer.pad(features_token_type, padding=True, return_tensors="pt")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "token_type_ids is not supported" in str(e)


if __name__ == "__main__":
    unittest.main()
