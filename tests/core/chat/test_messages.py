"""
Tests for the chat messages module
"""

import unittest

import pytest
from transformers import AddedToken, AutoTokenizer

from axolotl.core.chat.format.chatml import format_message
from axolotl.core.chat.messages import ChatFormattedChats, Chats

from tests.hf_offline_utils import enable_hf_offline  # noqa


@pytest.fixture(scope="session", name="llama_tokenizer")
@enable_hf_offline
def llama_tokenizer_fixture():
    return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")


@pytest.fixture(scope="session", name="chatml_tokenizer")
def llama_tokenizer_w_chatml(llama_tokenizer):
    llama_tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken(
                "<|im_end|>", rstrip=False, lstrip=False, normalized=False
            )
        }
    )
    llama_tokenizer.add_tokens(
        [
            AddedToken("<|im_start|>", rstrip=False, lstrip=False, normalized=False),
        ]
    )

    return llama_tokenizer


@pytest.fixture(scope="session", name="chat_msgs")
def chat_msgs_fixture():
    return {
        "conversation": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "value": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": "What is today's stock price of Apple?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "value": {
                            "name": "get_date",
                            "arguments": {},
                        },
                    },
                    {
                        "type": "tool_call",
                        "value": {
                            "name": "get_stock_price",
                            "arguments": {"symbol": "AAPL"},
                        },
                    },
                ],
                "weight": 1,
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_response",
                        "value": {
                            "name": "get_date",
                            "content": {"date": "2024-09-09"},
                        },
                    },
                    {
                        "type": "tool_response",
                        "value": {
                            "name": "get_stock_price",
                            "content": {"symbol": "AAPL", "price": 123.45},
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "value": "The stock price of Apple is $123.45.\n",
                        "weight": 0,
                    },
                    {
                        "type": "text",
                        "value": "<reflection>The original query asked for today's stock price of Apple. This implies they also wanted the date included in the response.</reflection>",
                    },
                    {
                        "type": "text",
                        "value": "The stock price of Apple on September 9, 2024 is $123.45.",
                    },
                ],
                "weight": 1,
            },
        ]
    }


class TestMessagesCase:
    """
    Test cases for the chat messages module
    """

    def test_tool_call_stringify(self, chat_msgs):
        chat_msgs_as_obj = Chats(**chat_msgs)
        assert '{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}' == str(
            chat_msgs_as_obj.conversation[2].content[1].value
        )

    def test_chatml_formatted_wrapper(self, chat_msgs):
        chat_msg_formatted = ChatFormattedChats(**chat_msgs, formatter=format_message)
        target_chatml = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is today's stock price of Apple?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_date", "arguments": {}}
</tool_call>
<tool_call>
{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}
</tool_call>
<|im_end|>
<|im_start|>tool
<tool_response>
{"name": "get_date", "content": {"date": "2024-09-09"}}
</tool_response>
<tool_response>
{"name": "get_stock_price", "content": {"symbol": "AAPL", "price": 123.45}}
</tool_response>
<|im_end|>
<|im_start|>assistant
The stock price of Apple is $123.45.
<reflection>The original query asked for today's stock price of Apple. This implies they also wanted the date included in the response.</reflection>The stock price of Apple on September 9, 2024 is $123.45.<|im_end|>\n"""
        assert target_chatml == str(chat_msg_formatted)

    def test_chatml_formatting_tool_call(self, chat_msgs):
        chat_msgs_as_obj = Chats(**chat_msgs)
        target_chatml_turn2 = """<|im_start|>assistant\n<tool_call>\n{"name": "get_date", "arguments": {}}\n</tool_call>\n<tool_call>\n{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}\n</tool_call>\n<|im_end|>\n"""
        assert target_chatml_turn2 == str(
            format_message(chat_msgs_as_obj.conversation[2])
        )

    def test_train_labels(self, chatml_tokenizer, chat_msgs):
        chat_msg_formatted = ChatFormattedChats(**chat_msgs, formatter=format_message)
        tokenized = chat_msg_formatted.conversation[2].tokenized(chatml_tokenizer)
        # fmt: off
        target_labels = [
            -100, -100, -100,  # role
            27, 14506, 13735, 397, 5018, 609, 794,
            330, 456, 4257, 498, 330, 16774, 794, 4792, 534, 524,
            14506, 13735, 397, 27, 14506, 13735, 397, 5018, 609, 794,
            330, 456, 31641, 9217, 498, 330, 16774, 794, 5324, 19314,
            794, 330, 84016, 43, 96742, 524, 14506, 13735, 397,
            128256,  # <|im_end|>
            -100  # trailing newline
        ]
        # fmt: on
        assert tokenized["labels"] == target_labels

    def test_train_labels_2(self, chatml_tokenizer, chat_msgs):
        # also test if indivudal contents are set not to train
        chat_msg_formatted = ChatFormattedChats(**chat_msgs, formatter=format_message)
        tokenized = chat_msg_formatted.conversation[4].tokenized(chatml_tokenizer)
        # fmt: off
        target_labels = [
            -100, -100, -100,  # role
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # initial response
            27, 78098, 16761, 4113, 3319, 4691, 369, 3432, 596, 5708, 3430,
            315, 8325, 13, 1115, 24897, 814, 1101, 4934, 279, 2457,
            5343, 304, 279, 2077, 4005, 78098, 16761, 5708, 3430, 315,
            8325, 389, 6250, 220, 24, 11, 220, 2366, 19, 374, 400,
            4513, 13, 1774, 13,
            128256,  # <|im_end|>
            -100,  # trailing newline
        ]
        # fmt: on
        assert tokenized["labels"] == target_labels


if __name__ == "__main__":
    unittest.main()
