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


def test_template_requiring_thinking_field_trains_reasoning(qwen3_tokenizer):
    chat_template_jinja = (
        "{%- for message in messages -%}"
        "{%- if message.role == 'assistant' -%}"
        "{%- if message.reasoning_content is not defined -%}"
        "{{ raise_exception('Assistant message is missing a thinking field.') }}"
        "{%- endif -%}"
        "{{ '<|im_start|>assistant\\n<think>' + message.reasoning_content + '</think>'"
        " + message.content + '<|im_end|>\\n' }}"
        "{%- else -%}"
        "{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- endif -%}"
        "{%- endfor -%}"
    )
    strategy = load(
        qwen3_tokenizer,
        DictDefault({"train_on_inputs": False, "sequence_len": 512}),
        DictDefault(
            {
                "chat_template": "jinja",
                "chat_template_jinja": chat_template_jinja,
                "field_messages": "messages",
            }
        ),
    )
    res = strategy.tokenize_prompt(
        {
            "messages": [
                {"role": "user", "content": "q0"},
                {"role": "assistant", "reasoning_content": "r0", "content": "a0"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
            ]
        }
    )
    trained = qwen3_tokenizer.decode(
        [
            i
            for i, label in zip(res["input_ids"], res["labels"], strict=True)
            if label != -100
        ]
    )
    assert trained == "r0</think>a0<|im_end|>r1</think>a1<|im_end|>"
