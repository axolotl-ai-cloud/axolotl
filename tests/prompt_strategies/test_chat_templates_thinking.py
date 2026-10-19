"""
Tests for splitting reasoning/thinking from content into separate field
"""

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import (
    load,
)
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="messages_w_reasoning")
def messages_w_reasoning_fixture():
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
                        "content": "<think>lorem</think>\nwelcome",
                    },
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                    },
                    {
                        "role": "assistant",
                        "content": "<|begin_of_thought|>lorem<|end_of_thought|>\n<|begin_of_solution|>welcome\n<|end_of_solution|>",
                    },
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                    },
                    {
                        "role": "assistant",
                        "content": "<reasoning>lorem</reasoning>\nwelcome",
                    },
                ]
            },
        ]
    )


@pytest.fixture(name="messages_w_tool_call")
def messages_w_tool_call_fixture():
    """An assistant tool call turn, which carries no content.

    ``content: null`` is what the OpenAI format emits for a tool call, and the
    key is absent altogether once the message property mapping skips the None.
    """
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "what is the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": "22C, sunny"},
                    {
                        "role": "assistant",
                        "content": "<think>lorem</think>\nwelcome",
                    },
                ]
            }
        ]
    )


class TestSplitThinking:
    """
    test class to make sure datasets with reasoning content conforms to the chat_template strategy
    """

    def test_splits_think(self, messages_w_reasoning, qwen3_tokenizer):
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
                    "split_thinking": True,
                }
            ),
        )
        for conversation in messages_w_reasoning:
            transformed_prompt = strategy.get_conversation_thread(conversation)
            assert transformed_prompt[0]["role"] == "user"
            assert transformed_prompt[1]["role"] == "assistant"
            assert transformed_prompt[1]["reasoning_content"] == "lorem"
            assert transformed_prompt[1]["content"] == "welcome"

            res = strategy.tokenize_prompt(conversation)
            input_ids = res["input_ids"]
            # fmt: off
            expected_input_ids = [
                151644,  # im_start
                872,  # user
                198,  # \n
                14990,  # hello
                151645,  # im_end
                198,  # \n
                151644,  # im_start
                77091,  # assistant
                198,  # \n
                151667,  # think
                198,  # \n
                385, 1826,  # lorem
                198,  # \n
                151668,  # /think
                271,  # \n
                34084,  # welcome
                151645,  # im_end
                198,  # \n
            ]
            # fmt: on
            assert input_ids == expected_input_ids, (
                f"Input IDs mismatch: {input_ids} != {expected_input_ids}"
            )

    def test_tool_call_turn_without_content(
        self, messages_w_tool_call, qwen3_tokenizer
    ):
        """A contentless assistant turn must not break the thinking split.

        The split reads the assistant content, and an OpenAI tool call turn has
        none, so the same conversation tokenizes with split_thinking off and
        raised KeyError with it on.
        """
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
                        "tool": ["tool"],
                    },
                    "field_messages": "messages",
                    "split_thinking": True,
                }
            ),
        )

        conversation = messages_w_tool_call[0]
        thread = strategy.get_conversation_thread(conversation)

        # the tool call survives, and carries no content to split
        assert thread[1]["role"] == "assistant"
        assert "content" not in thread[1]
        assert thread[1]["tool_calls"][0]["function"]["name"] == "get_weather"

        # the later assistant turn still splits normally
        assert thread[3]["reasoning_content"] == "lorem"
        assert thread[3]["content"] == "welcome"

        assert len(strategy.tokenize_prompt(conversation)["input_ids"]) > 0
