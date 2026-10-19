"""Tool results named from their call id for templates that resolve the sender via a registry."""

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault

# Harmony-style wire: a tool result renders its sender, found from `name` or from the
# ids of earlier tool calls, and raises otherwise (as Reflection Beam's template does).
REGISTRY_TEMPLATE = """{%- set registry = namespace(entries=[]) -%}
{%- for message in messages -%}
{%- if message.role == 'user' -%}
<|start|>user<|message|>{{ message.content }}<|end|>
{%- elif message.role == 'assistant' -%}
<|start|>assistant<|channel|>analysis<|message|>{{ message.reasoning_content or '' }}<|end|>
{%- if message.tool_calls -%}
{%- for call in message.tool_calls -%}
{%- set registry.entries = registry.entries + [{"id": call.id, "name": call.function.name}] -%}
<|start|>assistant to={{ call.function.name }}<|channel|>analysis<|message|>{{ call.function.arguments | tojson }}<|call|>
{%- endfor -%}
{%- else -%}
<|start|>assistant<|channel|>final<|message|>{{ message.content }}<|return|>
{%- endif -%}
{%- elif message.role == 'tool' -%}
{%- set sender = namespace(name=none) -%}
{%- if message.name is defined and message.name -%}
{%- set sender.name = message.name -%}
{%- elif message.tool_call_id is defined -%}
{%- for entry in registry.entries -%}
{%- if entry.id == message.tool_call_id -%}{%- set sender.name = entry.name -%}{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- if sender.name is none -%}
{{ raise_exception("Tool message needs 'name' or a matching prior tool_call id") }}
{%- endif -%}
<|start|>{{ sender.name }} to=assistant<|channel|>analysis<|message|>{{ message.content }}<|end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}
"""


def _registry_strategy(tokenizer, roles_to_train):
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|start|>",
                "<|end|>",
                "<|message|>",
                "<|channel|>",
                "<|call|>",
                "<|return|>",
            ]
        }
    )
    # Harmony-style tokenizers end the final channel with the EOS token itself
    tokenizer.eos_token = "<|return|>"
    return load(
        tokenizer,
        DictDefault({"train_on_inputs": False, "sequence_len": 1024}),
        DictDefault(
            {
                "chat_template": "jinja",
                "chat_template_jinja": REGISTRY_TEMPLATE,
                "eot_tokens": ["<|end|>", "<|call|>", "<|return|>"],
                "field_messages": "messages",
                "roles_to_train": roles_to_train,
            }
        ),
    )


@pytest.fixture(name="registry_strategy")
def fixture_registry_strategy(smollm2_tokenizer):
    return _registry_strategy(smollm2_tokenizer, ["assistant"])


@pytest.fixture(name="tool_result_by_id_dataset")
def fixture_tool_result_by_id_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Which host raised alert A1?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Look the alert up first.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "get_alert",
                                    "arguments": {"alert_id": "A1"},
                                },
                            }
                        ],
                    },
                    # standard OpenAI shape: the result carries the call id, not the tool name
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": '{"host": "ws-101"}',
                    },
                    {
                        "role": "assistant",
                        "content": "Alert A1 came from ws-101.",
                        "reasoning_content": "The record names the host.",
                    },
                ]
            }
        ]
    )


def test_tool_result_named_from_call_id(
    registry_strategy, tool_result_by_id_dataset, smollm2_tokenizer
):
    tokenizer = smollm2_tokenizer
    processed = tool_result_by_id_dataset.map(
        registry_strategy.tokenize_prompt, remove_columns=["messages"]
    )
    ids, labels = processed[0]["input_ids"], processed[0]["labels"]

    text = tokenizer.decode(ids)
    assert "<|start|>get_alert to=assistant" in text
    trained = tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l != -100]
    )
    masked = tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l == -100]
    )
    # the call and both finals are trained; the user turn and the tool result are not
    assert (
        "get_alert" in trained
        and '"alert_id": "A1"' in trained
        and "<|call|>" in trained
    )
    assert "Alert A1 came from ws-101.<|return|>" in trained
    assert '{"host": "ws-101"}' in masked and "Which host raised alert A1?" in masked
    assert '{"host": "ws-101"}' not in trained


def test_trainable_tool_result_keeps_sender_in_placeholder(
    smollm2_tokenizer, tool_result_by_id_dataset
):
    tokenizer = smollm2_tokenizer
    # a trainable tool result is located through a placeholder of itself
    strategy = _registry_strategy(tokenizer, ["assistant", "tool"])
    processed = tool_result_by_id_dataset.map(
        strategy.tokenize_prompt, remove_columns=["messages"]
    )
    ids, labels = processed[0]["input_ids"], processed[0]["labels"]

    trained = tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l != -100]
    )
    masked = tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l == -100]
    )
    assert '{"host": "ws-101"}' in trained
    assert "Which host raised alert A1?" in masked
    assert "Which host raised alert A1?" not in trained


def test_tool_result_without_any_resolution_still_raises(
    registry_strategy, smollm2_tokenizer
):
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "missing", "content": "{}"},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        ]
    )
    with pytest.raises(Exception, match="matching prior tool_call"):
        dataset.map(registry_strategy.tokenize_prompt, remove_columns=["messages"])
