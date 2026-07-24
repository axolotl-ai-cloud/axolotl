"""
tests for chat_template prompt strategy
"""

import json
import unittest

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.prompt_strategies.dpo.chat_template import argilla_chat, default
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="assistant_dataset")
def fixture_assistant_dataset():
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


@pytest.fixture(name="argilla_chat_dataset")
def fixture_argilla_chat_dataset():
    return Dataset.from_list(
        [
            {
                "chosen": [
                    {
                        "role": "user",
                        "content": "hello",
                    },
                    {
                        "role": "assistant",
                        "content": "goodbye",
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "hello",
                    },
                    {
                        "role": "assistant",
                        "content": "party on",
                    },
                ],
            }
        ]
    )


@pytest.fixture(name="toolcalling_dpo_dataset")
def fixture_toolcalling_dpo_dataset():
    """OpenAI-format tool calling DPO dataset with tools and tool_calls."""
    return Dataset.from_list(
        [
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_temperature",
                            "description": "Get the current temperature in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the temperature in Paris?",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_current_temperature",
                                    "arguments": '{"location": "Paris, France"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "22.0",
                    },
                ],
                "chosen": {
                    "role": "assistant",
                    "content": "The temperature in Paris is 22.0 degrees Celsius.",
                },
                "rejected": {
                    "role": "assistant",
                    "content": "I don't know.",
                },
            }
        ]
    )


@pytest.fixture(name="phi3_tokenizer")
@enable_hf_offline
def fixture_phi3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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
        transform_fn, _ = default(
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
        transform_fn, _ = default(
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

    @pytest.mark.xfail(reason="likely upstream issue from v5.4.0")
    def test_phi3_defaults(self, phi3_tokenizer, assistant_dataset):
        transform_fn, _ = default(
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
        assert result["chosen"] == "goodbye<|end|>\n<|endoftext|>"
        assert result["rejected"] == "party on<|end|>\n<|endoftext|>"


class TestAssistantDPOChatTemplateGemma:
    """
    Test class for assistant style datasets with gemma prompts using the tokenizer's chat_template strategy.
    """

    def test_gemma_defaults(self, gemma_tokenizer, assistant_dataset):
        transform_fn, _ = default(
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


class TestArgillaChatDPOChatTemplate:
    """
    Test class for argilla_chat style datasets (chosen/rejected contain full conversations).
    """

    def test_llama3_argilla_chat(self, llama3_tokenizer, argilla_chat_dataset):
        transform_fn, _ = argilla_chat(
            DictDefault(
                {
                    "chat_template": "llama3",
                    "datasets": [
                        {
                            "type": "chat_template.argilla_chat",
                        }
                    ],
                }
            )
        )
        result = transform_fn(argilla_chat_dataset[0], tokenizer=llama3_tokenizer)
        assert result["prompt"] == (
            "<|begin_of_text|>"
            + "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert result["chosen"] == "goodbye<|eot_id|>"
        assert result["rejected"] == "party on<|eot_id|>"

    @pytest.mark.xfail(reason="likely upstream issue from v5.4.0")
    def test_phi3_argilla_chat(self, phi3_tokenizer, argilla_chat_dataset):
        transform_fn, _ = argilla_chat(
            DictDefault(
                {
                    "chat_template": "tokenizer_default",
                    "datasets": [
                        {
                            "type": "chat_template.argilla_chat",
                        }
                    ],
                }
            )
        )
        result = transform_fn(argilla_chat_dataset[0], tokenizer=phi3_tokenizer)
        assert result["prompt"] == "<|user|>\nhello<|end|>\n" + "<|assistant|>\n"
        assert result["chosen"] == "goodbye<|end|>\n<|endoftext|>"
        assert result["rejected"] == "party on<|end|>\n<|endoftext|>"


class TestDPOChatTemplateToolRole:
    """
    Test that DPO chat template strategy handles tool role messages without KeyError.
    Regression test for https://github.com/axolotl-ai-cloud/axolotl/issues/3217
    """

    def test_tool_role_default_no_key_error(self, llama3_tokenizer):
        """Messages list with a 'tool' role should not raise KeyError."""
        dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "What is the weather?"},
                        {
                            "role": "assistant",
                            "content": "Let me check.",
                        },
                        {
                            "role": "tool",
                            "content": "22°C, sunny.",
                        },
                    ],
                    "chosen": {
                        "role": "assistant",
                        "content": "It is 22°C and sunny.",
                    },
                    "rejected": {
                        "role": "assistant",
                        "content": "I don't know.",
                    },
                }
            ]
        )
        transform_fn, _ = default(
            DictDefault(
                {
                    "chat_template": "llama3",
                    "datasets": [{"type": "chat_template"}],
                }
            )
        )
        # Should not raise KeyError: 'tool'
        result = transform_fn(dataset[0], tokenizer=llama3_tokenizer)
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result

    def test_tool_role_custom_mapping_preserved(self, llama3_tokenizer):
        """A user-supplied roles mapping that overrides 'tool' is still respected."""
        dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "tool_result", "content": "42"},
                    ],
                    "chosen": {"role": "assistant", "content": "The answer is 42."},
                    "rejected": {"role": "assistant", "content": "Unknown."},
                }
            ]
        )
        transform_fn, _ = default(
            DictDefault(
                {
                    "chat_template": "llama3",
                    "datasets": [
                        {
                            "type": "chat_template",
                            "roles": {
                                "user": ["user"],
                                "assistant": ["assistant"],
                                "system": ["system"],
                                "tool": ["tool_result"],
                            },
                        }
                    ],
                }
            )
        )
        result = transform_fn(dataset[0], tokenizer=llama3_tokenizer)
        assert "prompt" in result


class TestDPOChatTemplateToolCalling:
    """
    Test class for OpenAI-style tool calling datasets with the DPO chat_template strategy.
    """

    def test_tools_and_tool_calls_rendered(
        self, llama3_tokenizer, toolcalling_dpo_dataset
    ):
        """tools render into the system prompt and tool_calls are preserved."""
        transform_fn, ds_kwargs = default(
            DictDefault(
                {
                    "chat_template": "qwen_25",
                    "datasets": [{"type": "chat_template"}],
                }
            )
        )
        assert "tools" in ds_kwargs["remove_columns"]

        result = transform_fn(toolcalling_dpo_dataset[0], tokenizer=llama3_tokenizer)
        prompt = result["prompt"]

        # tool definitions are rendered into the system prompt
        assert "<tools>" in prompt
        assert '"name": "get_current_temperature"' in prompt

        # assistant tool call is preserved and JSON string arguments are decoded
        assert (
            '<tool_call>\n{"name": "get_current_temperature", '
            '"arguments": {"location": "Paris, France"}}\n</tool_call>' in prompt
        )

        # tool response turn is rendered
        assert "<tool_response>\n22.0\n</tool_response>" in prompt

        assert prompt.endswith("<|im_start|>assistant\n")
        assert (
            result["chosen"]
            == "The temperature in Paris is 22.0 degrees Celsius.<|im_end|>"
        )
        assert result["rejected"] == "I don't know.<|im_end|>"

    def test_tool_call_as_response(self, llama3_tokenizer, toolcalling_dpo_dataset):
        """chosen/rejected can themselves be tool call messages without content."""
        sample = dict(toolcalling_dpo_dataset[0])
        sample["messages"] = sample["messages"][:1]
        sample["chosen"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_temperature",
                        "arguments": '{"location": "Paris, France"}',
                    },
                }
            ],
        }
        sample["rejected"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_temperature",
                        "arguments": '{"location": "London, UK"}',
                    },
                }
            ],
        }

        transform_fn, _ = default(
            DictDefault(
                {
                    "chat_template": "qwen_25",
                    "datasets": [{"type": "chat_template"}],
                }
            )
        )
        result = transform_fn(sample, tokenizer=llama3_tokenizer)

        assert result["chosen"] == (
            '<tool_call>\n{"name": "get_current_temperature", '
            '"arguments": {"location": "Paris, France"}}\n</tool_call><|im_end|>'
        )
        assert result["rejected"] == (
            '<tool_call>\n{"name": "get_current_temperature", '
            '"arguments": {"location": "London, UK"}}\n</tool_call><|im_end|>'
        )
        assert "[[dummy_message]]" not in result["chosen"]
        assert "[[dummy_message]]" not in result["rejected"]

    def test_tools_as_json_strings(self, llama3_tokenizer, toolcalling_dpo_dataset):
        """tools stored as a list of JSON-encoded strings are decoded before rendering."""
        sample = dict(toolcalling_dpo_dataset[0])
        sample["tools"] = [json.dumps(tool) for tool in sample["tools"]]

        transform_fn, _ = default(
            DictDefault(
                {
                    "chat_template": "qwen_25",
                    "datasets": [{"type": "chat_template"}],
                }
            )
        )
        result = transform_fn(sample, tokenizer=llama3_tokenizer)

        assert "<tools>" in result["prompt"]
        assert '"name": "get_current_temperature"' in result["prompt"]


class TestDPOChatTemplateReasoning:
    """
    Test class for reasoning (thinking) responses with the DPO chat_template strategy.
    """

    def test_reasoning_content_preserved(self, llama3_tokenizer):
        """reasoning_content in chosen/rejected renders into the think block."""
        sample = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "chosen": {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "Simple arithmetic: 2+2 equals 4.",
            },
            "rejected": {
                "role": "assistant",
                "content": "The answer is 5.",
                "reasoning_content": "2+2 is 5 because I said so.",
            },
        }

        transform_fn, _ = default(
            DictDefault(
                {
                    "chat_template": "qwen3_5",
                    "datasets": [{"type": "chat_template"}],
                }
            )
        )
        result = transform_fn(sample, tokenizer=llama3_tokenizer)

        # the generation prompt opens the think block; responses continue it
        assert result["prompt"].endswith("<|im_start|>assistant\n<think>\n\n")
        assert result["chosen"] == (
            "Simple arithmetic: 2+2 equals 4.\n</think>\n\nThe answer is 4.<|im_end|>"
        )
        assert result["rejected"] == (
            "2+2 is 5 because I said so.\n</think>\n\nThe answer is 5.<|im_end|>"
        )

    def test_chat_template_kwargs_passed(self, llama3_tokenizer):
        """chat_template_kwargs (e.g. enable_thinking) reach apply_chat_template."""
        sample = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "chosen": {"role": "assistant", "content": "The answer is 4."},
            "rejected": {"role": "assistant", "content": "The answer is 5."},
        }

        transform_fn, _ = default(
            DictDefault(
                {
                    "chat_template": "qwen3_5",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "datasets": [{"type": "chat_template"}],
                }
            )
        )
        result = transform_fn(sample, tokenizer=llama3_tokenizer)

        # with thinking disabled the generation prompt closes the think block
        assert result["prompt"].endswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        assert result["chosen"] == "The answer is 4.<|im_end|>"
        assert result["rejected"] == "The answer is 5.<|im_end|>"


if __name__ == "__main__":
    unittest.main()
