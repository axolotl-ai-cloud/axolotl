"""Labeling assistant turns from a template's ``{% generation %}`` markers."""

import copy
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault

# Harmony-style template with the whole assistant output inside generation markers, so
# channel headers and the empty analysis block are generated, not just field contents.
GENERATION_TEMPLATE = """{%- for message in messages -%}
{%- if message.role == 'user' -%}
<|start|>user<|message|>{{ message.content }}<|end|>
{%- elif message.role == 'assistant' -%}
<|start|>assistant{% generation %}<|channel|>analysis<|message|>{{ message.reasoning_content or '' }}<|end|>
{%- if message.tool_calls -%}
{%- for call in message.tool_calls -%}
<|start|>assistant to={{ call.function.name }}<|channel|>analysis<|message|>{{ call.function.arguments | tojson }}<|call|>
{%- endfor -%}
{%- else -%}
<|start|>assistant<|channel|>final<|message|>{{ message.content }}<|return|>
{%- endif -%}
{% endgeneration %}
{%- elif message.role == 'tool' -%}
<|start|>{{ message.name }} to=assistant<|channel|>analysis<|message|>{{ message.content }}<|end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}
"""

NO_MARKER_TEMPLATE = GENERATION_TEMPLATE.replace("{% generation %}", "").replace(
    "{% endgeneration %}", ""
)

SPECIALS = [
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|call|>",
    "<|return|>",
]


@pytest.fixture(name="smollm2_tokenizer")
def fixture_smollm2_tokenizer(smollm2_tokenizer):
    # _strategy adds special tokens and swaps eos_token; keep the session tokenizer clean
    return copy.deepcopy(smollm2_tokenizer)


def _strategy(
    tokenizer,
    template=GENERATION_TEMPLATE,
    eot_tokens=("<|end|>", "<|call|>", "<|return|>"),
    **ds_overrides,
):
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})
    tokenizer.eos_token = "<|return|>"
    ds_cfg = {
        "chat_template": "jinja",
        "chat_template_jinja": template,
        "field_messages": "messages",
        "roles_to_train": ["assistant"],
        "message_field_training": "train",
        "generation_tags": True,
    }
    ds_cfg.update(ds_overrides)
    return load(
        tokenizer,
        DictDefault(
            {
                "train_on_inputs": False,
                "sequence_len": 1024,
                "eot_tokens": list(eot_tokens),
            }
        ),
        DictDefault(ds_cfg),
    )


def _conversation(train_last=None):
    final = {
        "role": "assistant",
        "content": "Alert A1 came from ws-101.",
        "reasoning_content": "The record names the host.",
    }
    if train_last is not None:
        final["train"] = train_last
    return {
        "messages": [
            {"role": "user", "content": "Which host raised alert A1?"},
            # no reasoning: the model still emits the empty analysis block and its close
            {
                "role": "assistant",
                "content": "",
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
            {
                "role": "tool",
                "name": "get_alert",
                "tool_call_id": "call-1",
                "content": '{"host": "ws-101"}',
            },
            final,
        ]
    }


def _split(tokenizer, row):
    ids, labels = row["input_ids"], row["labels"]
    trained = tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l != -100]
    )
    masked = tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l == -100]
    )
    return ids, labels, trained, masked


def test_structural_tokens_are_labeled(smollm2_tokenizer):
    strategy = _strategy(smollm2_tokenizer)
    row = Dataset.from_list([_conversation()]).map(strategy.tokenize_prompt)[0]
    ids, labels, trained, masked = _split(smollm2_tokenizer, row)

    full = smollm2_tokenizer.decode(ids)
    assert full == smollm2_tokenizer.decode(row["input_ids"])
    # the empty analysis block, its close and the handoff into the tool call are generated
    assert (
        "<|channel|>analysis<|message|><|end|><|start|>assistant to=get_alert<|channel|>analysis<|message|>"
        in trained
    )
    assert '"alert_id": "A1"' in trained and "<|call|>" in trained
    # the channel switch from analysis to final is generated too
    assert (
        "The record names the host.<|end|><|start|>assistant<|channel|>final<|message|>Alert A1 came from ws-101.<|return|>"
        in trained
    )
    # the prompt and the tool result are not, nor is the assistant header the prompt supplies
    assert "Which host raised alert A1?" in masked
    assert '{"host": "ws-101"}' in masked and "get_alert to=assistant" in masked
    assert (
        trained.count("<|start|>assistant<|channel|>") == 1
    )  # only the one inside a block
    assert all(l == -100 or l == t for t, l in zip(ids, labels, strict=True))


def test_per_turn_train_flag_masks_a_block(smollm2_tokenizer):
    strategy = _strategy(smollm2_tokenizer)
    row = Dataset.from_list([_conversation(train_last=False)]).map(
        strategy.tokenize_prompt
    )[0]
    _, _, trained, masked = _split(smollm2_tokenizer, row)
    assert "<|call|>" in trained
    assert "Alert A1 came from ws-101." in masked and "<|return|>" in masked


@pytest.mark.parametrize(
    "train_on_eot, expect_call, expect_return",
    [("turn", True, True), ("last", False, True), ("none", False, False)],
)
def test_train_on_eot_applies_inside_blocks(
    smollm2_tokenizer, train_on_eot, expect_call, expect_return
):
    strategy = _strategy(
        smollm2_tokenizer, train_on_eot=train_on_eot, train_on_eos=train_on_eot
    )
    row = Dataset.from_list([_conversation()]).map(strategy.tokenize_prompt)[0]
    ids, labels = row["input_ids"], row["labels"]
    call_id = smollm2_tokenizer.convert_tokens_to_ids("<|call|>")
    return_id = smollm2_tokenizer.convert_tokens_to_ids("<|return|>")
    assert any(l == call_id for l in labels) is expect_call
    assert any(l == return_id for l in labels) is expect_return
    # everything else inside the blocks stays labeled
    assert '"alert_id": "A1"' in smollm2_tokenizer.decode(
        [t for t, l in zip(ids, labels, strict=True) if l != -100]
    )


def test_requires_generation_markers(smollm2_tokenizer):
    with pytest.raises(ValueError, match="generation"):
        _strategy(smollm2_tokenizer, template=NO_MARKER_TEMPLATE)


def test_matches_diff_locator_on_field_content(smollm2_tokenizer):
    tagged = _strategy(smollm2_tokenizer)
    plain = _strategy(smollm2_tokenizer, generation_tags=None)
    data = Dataset.from_list([_conversation()])
    tagged_row = data.map(tagged.tokenize_prompt)[0]
    plain_row = data.map(plain.tokenize_prompt)[0]
    assert tagged_row["input_ids"] == plain_row["input_ids"]
    # every token the diff locator trains, the marker path trains too; the marker path
    # adds only template structure the model has to generate
    extra = [
        t
        for t, a, b in zip(
            tagged_row["input_ids"],
            tagged_row["labels"],
            plain_row["labels"],
            strict=True,
        )
        if a != -100 and b == -100
    ]
    assert not [
        t
        for a, b in zip(tagged_row["labels"], plain_row["labels"], strict=True)
        if b != -100 and a == -100
    ]
    assert set(smollm2_tokenizer.convert_ids_to_tokens(extra)) <= set(
        SPECIALS + ["analysis", "final", "assistant", "Ġto", "=", "get", "_alert"]
    ) | set(smollm2_tokenizer.tokenize("assistant to=get_alertanalysisfinal"))


@pytest.mark.parametrize(
    "train_on_eos, expect_return",
    [("turn", True), ("last", True), ("none", False), ("all", True)],
)
def test_train_on_eos_applies_when_eos_is_not_an_eot_token(
    smollm2_tokenizer, train_on_eos, expect_return
):
    strategy = _strategy(
        smollm2_tokenizer,
        eot_tokens=("<|end|>", "<|call|>"),
        train_on_eot="turn",
        train_on_eos=train_on_eos,
    )
    row = Dataset.from_list([_conversation()]).map(strategy.tokenize_prompt)[0]
    labels = row["labels"]
    call_id = smollm2_tokenizer.convert_tokens_to_ids("<|call|>")
    return_id = smollm2_tokenizer.convert_tokens_to_ids("<|return|>")
    assert any(l == return_id for l in labels) is expect_return
    assert any(l == call_id for l in labels)


def test_train_on_eos_last_keeps_only_the_last_blocks_eos(smollm2_tokenizer):
    conversation = _conversation()
    # a second final answer before the tool turn puts an EOS in an earlier block
    conversation["messages"][1:1] = [
        {"role": "assistant", "content": "Looking it up."},
        {"role": "user", "content": "Go ahead."},
    ]
    strategy = _strategy(
        smollm2_tokenizer,
        eot_tokens=("<|end|>", "<|call|>"),
        train_on_eot="turn",
        train_on_eos="last",
    )
    row = Dataset.from_list([conversation]).map(strategy.tokenize_prompt)[0]
    return_id = smollm2_tokenizer.convert_tokens_to_ids("<|return|>")
    assert row["input_ids"].count(return_id) == 2
    assert row["labels"].count(return_id) == 1
    assert (
        row["labels"][len(row["labels"]) - 1 - row["input_ids"][::-1].index(return_id)]
        == return_id
    )


def test_mistral_strategy_rejects_generation_tags():
    from axolotl.prompt_strategies.chat_template import MistralStrategy

    with pytest.raises(ValueError, match="generation_tags"):
        MistralStrategy(
            MagicMock(),
            tokenizer=MagicMock(),
            train_on_inputs=False,
            sequence_len=512,
            generation_tags=True,
        )
