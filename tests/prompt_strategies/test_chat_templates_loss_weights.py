"""Per-message and per-row loss weights in the chat_template strategy."""

from copy import deepcopy

import pytest

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.prompters import IGNORE_TOKEN_ID


def _strategy(tokenizer, roles_to_train=("assistant",), train_on_eos="turn", **kw):
    prompter = ChatTemplatePrompter(
        tokenizer,
        chat_template=tokenizer.chat_template,
        message_field_training="training",
        **kw,
    )
    return ChatTemplateStrategy(
        prompter,
        tokenizer=tokenizer,
        train_on_inputs=False,
        sequence_len=2048,
        roles_to_train=list(roles_to_train) if roles_to_train else None,
        train_on_eos=train_on_eos,
        eot_tokens=["<|im_end|>"],
    )


def _trained(labels):
    return {i for i, label in enumerate(labels) if label != IGNORE_TOKEN_ID}


def _only_turn_trained(tokenizer, row, keep):
    """Labels when only message ``keep`` is trainable (others ``training: false``)."""
    row = deepcopy(row)
    for i, message in enumerate(row["messages"]):
        if message["role"] == "assistant" and i != keep:
            message["training"] = False
    return _trained(_strategy(tokenizer).tokenize_prompt(row)["labels"])


@pytest.fixture(name="multiturn_row")
def fixture_multiturn_row():
    return {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4.", "w": 0.25},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "It is 6.", "w": 2.0},
            {"role": "user", "content": "Thanks"},
            {
                "role": "assistant",
                "content": "You're welcome.",
                "reasoning_content": "The user said thanks.",
            },
        ]
    }


@pytest.fixture(name="toolcall_row")
def fixture_toolcall_row():
    return {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Paris"},
                        },
                    }
                ],
                "w": 0.5,
            },
            {"role": "tool", "name": "get_weather", "content": "22C"},
            {"role": "assistant", "content": "It is 22C in Paris.", "w": 1.5},
        ]
    }


class TestFeatureOff:
    def test_no_loss_weights_column_by_default(self, qwen3_tokenizer, multiturn_row):
        out = _strategy(qwen3_tokenizer).tokenize_prompt(deepcopy(multiturn_row))
        assert "loss_weights" not in out

    def test_enabled_without_values_matches_unweighted(
        self, qwen3_tokenizer, multiturn_row
    ):
        row = deepcopy(multiturn_row)
        for message in row["messages"]:
            message.pop("w", None)
        base = _strategy(qwen3_tokenizer).tokenize_prompt(deepcopy(row))
        out = _strategy(
            qwen3_tokenizer, message_field_weight="w", field_sample_weight="sw"
        ).tokenize_prompt(deepcopy(row))
        assert out["input_ids"] == base["input_ids"]
        assert out["labels"] == base["labels"]
        assert out["loss_weights"] == [
            1.0 if label != IGNORE_TOKEN_ID else 0.0 for label in base["labels"]
        ]


class TestAlignment:
    @pytest.mark.parametrize("row_fixture", ["multiturn_row", "toolcall_row"])
    def test_weights_land_on_each_turns_trained_tokens(
        self, qwen3_tokenizer, row_fixture, request
    ):
        row = request.getfixturevalue(row_fixture)
        base = _strategy(qwen3_tokenizer).tokenize_prompt(deepcopy(row))
        out = _strategy(qwen3_tokenizer, message_field_weight="w").tokenize_prompt(
            deepcopy(row)
        )

        assert out["input_ids"] == base["input_ids"]
        assert out["labels"] == base["labels"]
        assert len(out["loss_weights"]) == len(out["labels"])

        covered = set()
        for idx, message in enumerate(row["messages"]):
            if message["role"] != "assistant":
                continue
            span = _only_turn_trained(qwen3_tokenizer, row, idx)
            assert span, f"turn {idx} trained nothing"
            expected = message.get("w", 1.0)
            assert {out["loss_weights"][i] for i in span} == {expected}
            covered |= span

        assert covered == _trained(out["labels"])
        assert all(
            out["loss_weights"][i] == 0.0
            for i in range(len(out["labels"]))
            if i not in covered
        )

    def test_reasoning_content_tokens_get_turn_weight(
        self, qwen3_tokenizer, multiturn_row
    ):
        row = deepcopy(multiturn_row)
        row["messages"][-1]["w"] = 3.0
        out = _strategy(qwen3_tokenizer, message_field_weight="w").tokenize_prompt(row)
        ids = out["input_ids"]
        think_start = ids.index(qwen3_tokenizer.convert_tokens_to_ids("<think>"))
        think_end = ids.index(qwen3_tokenizer.convert_tokens_to_ids("</think>"))
        trained = [
            i
            for i in range(think_start + 1, think_end)
            if out["labels"][i] != IGNORE_TOKEN_ID
        ]
        assert "The user said thanks." in qwen3_tokenizer.decode(
            [ids[i] for i in trained]
        )
        assert {out["loss_weights"][i] for i in trained} == {3.0}

    def test_training_detail_composes(self, qwen3_tokenizer):
        row = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me think.", "train": False},
                        {"text": " The answer is 4.", "train": True},
                    ],
                    "w": 0.5,
                },
            ]
        }
        base = _strategy(qwen3_tokenizer).tokenize_prompt(deepcopy(row))
        out = _strategy(qwen3_tokenizer, message_field_weight="w").tokenize_prompt(
            deepcopy(row)
        )
        assert out["labels"] == base["labels"]
        assert {out["loss_weights"][i] for i in _trained(base["labels"])} == {0.5}


class TestZeroWeight:
    def test_zero_message_weight_equals_training_false(
        self, qwen3_tokenizer, multiturn_row
    ):
        masked = deepcopy(multiturn_row)
        masked["messages"][4]["training"] = False
        expected = _strategy(qwen3_tokenizer).tokenize_prompt(masked)

        weighted = deepcopy(multiturn_row)
        weighted["messages"][4]["w"] = 0
        out = _strategy(qwen3_tokenizer, message_field_weight="w").tokenize_prompt(
            weighted
        )

        assert out["labels"] == expected["labels"]
        assert all(
            (w == 0.0) == (label == IGNORE_TOKEN_ID)
            for w, label in zip(out["loss_weights"], out["labels"], strict=True)
        )

    def test_zero_row_weight_masks_everything(self, qwen3_tokenizer, multiturn_row):
        row = deepcopy(multiturn_row)
        row["sw"] = 0.0
        out = _strategy(
            qwen3_tokenizer, message_field_weight="w", field_sample_weight="sw"
        ).tokenize_prompt(row)
        assert set(out["labels"]) == {IGNORE_TOKEN_ID}
        assert set(out["loss_weights"]) == {0.0}


class TestRowWeight:
    def test_row_weight_multiplies_message_weight(self, qwen3_tokenizer, multiturn_row):
        message_only = _strategy(
            qwen3_tokenizer, message_field_weight="w"
        ).tokenize_prompt(deepcopy(multiturn_row))
        row = deepcopy(multiturn_row)
        row["sw"] = 4.0
        out = _strategy(
            qwen3_tokenizer, message_field_weight="w", field_sample_weight="sw"
        ).tokenize_prompt(row)
        assert out["labels"] == message_only["labels"]
        assert out["loss_weights"] == [4.0 * w for w in message_only["loss_weights"]]

    def test_row_weight_on_legacy_path(self, qwen3_tokenizer):
        row = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "sw": 0.5,
        }
        prompter = ChatTemplatePrompter(
            qwen3_tokenizer,
            chat_template=qwen3_tokenizer.chat_template,
            field_sample_weight="sw",
        )
        strategy = ChatTemplateStrategy(
            prompter,
            tokenizer=qwen3_tokenizer,
            train_on_inputs=False,
            sequence_len=2048,
        )
        out = strategy.tokenize_prompt(row)
        trained = _trained(out["labels"])
        assert trained
        assert out["loss_weights"] == [
            0.5 if i in trained else 0.0 for i in range(len(out["labels"]))
        ]

    def test_batched_rows(self, qwen3_tokenizer, multiturn_row):
        rows = [deepcopy(multiturn_row), deepcopy(multiturn_row)]
        rows[0]["sw"], rows[1]["sw"] = 1.0, 2.0
        batch = {key: [r[key] for r in rows] for key in rows[0]}
        out = _strategy(
            qwen3_tokenizer, message_field_weight="w", field_sample_weight="sw"
        ).tokenize_prompt(batch)
        assert out["loss_weights"][1] == [2.0 * w for w in out["loss_weights"][0]]


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "0.5", True])
def test_invalid_weight_raises(qwen3_tokenizer, multiturn_row, bad):
    row = deepcopy(multiturn_row)
    row["messages"][2]["w"] = bad
    with pytest.raises(ValueError, match="`w` must be"):
        _strategy(qwen3_tokenizer, message_field_weight="w").tokenize_prompt(row)


def test_strategy_loader_passes_weight_fields(qwen3_tokenizer, multiturn_row):
    from axolotl.prompt_strategies.chat_template import load
    from axolotl.utils.dict import DictDefault

    strategy = load(
        qwen3_tokenizer,
        DictDefault(train_on_inputs=False, sequence_len=2048),
        {
            "type": "chat_template",
            "chat_template": "tokenizer_default",
            "message_field_weight": "w",
            "field_sample_weight": "sw",
        },
    )
    row = deepcopy(multiturn_row)
    row["sw"] = 2.0
    out = strategy.tokenize_prompt(row)
    assert {w for w in out["loss_weights"] if w} == {0.5, 4.0, 2.0}
