"""Per-dataset and per-row chat template kwargs for the chat_template strategy."""

import json

import pyarrow as pa
import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import load
from axolotl.prompters import IGNORE_TOKEN_ID
from axolotl.utils.data.work_queue import tokenize_with_work_queue
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import SFTDataset

# Mirrors Beam's header: an int effort or a label, and a hard failure on numeric strings.
TEMPLATE = (
    "{%- if reasoning_effort is defined and reasoning_effort is string"
    " and reasoning_effort.isdigit() %}"
    "{{ raise_exception('reasoning_effort must be an int') }}"
    "{%- endif %}"
    "<|im_start|>system\nReasoning effort: {{ reasoning_effort | default(32) }}"
    "{% if enable_thinking is defined %}\nThinking: {{ enable_thinking }}{% endif %}"
    "{% if tools %}\nTools: {{ tools | map(attribute='function.name') | join(', ') }}{% endif %}"
    "<|im_end|>\n"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look something up",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

MESSAGES = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "That makes 6."},
]


def _strategy(tokenizer, global_kwargs=None, dataset_kwargs=None, **ds_overrides):
    cfg = DictDefault(
        {
            "sequence_len": 512,
            "train_on_inputs": False,
            "eot_tokens": ["<|im_end|>"],
            "chat_template_kwargs": global_kwargs,
        }
    )
    ds_cfg = {
        "type": "chat_template",
        "chat_template": "jinja",
        "chat_template_jinja": TEMPLATE,
        "field_chat_template_kwargs": "template_kwargs",
        "chat_template_kwargs": dataset_kwargs,
        **ds_overrides,
    }
    return load(tokenizer, cfg, ds_cfg)


def _row(template_kwargs="__missing__", **extra):
    row = {"messages": [dict(m) for m in MESSAGES], **extra}
    if template_kwargs != "__missing__":
        row["template_kwargs"] = template_kwargs
    return row


def _render(strategy, tokenizer, row):
    return tokenizer.decode(strategy.tokenize_prompt(row)["input_ids"])


def _trained_text(tokenizer, tokenized):
    return tokenizer.decode(
        [
            tok
            for tok, label in zip(
                tokenized["input_ids"], tokenized["labels"], strict=True
            )
            if label != IGNORE_TOKEN_ID
        ]
    )


@pytest.fixture(name="tokenizer")
def fixture_tokenizer(smollm2_tokenizer):
    return smollm2_tokenizer


class TestRowKwargsRendering:
    def test_rows_with_and_without_field(self, tokenizer):
        strategy = _strategy(tokenizer)
        assert "Reasoning effort: 128" in _render(
            strategy, tokenizer, _row({"reasoning_effort": 128})
        )
        for template_kwargs in ("__missing__", None, {}, ""):
            assert "Reasoning effort: 32" in _render(
                strategy, tokenizer, _row(template_kwargs)
            )

    def test_field_unset_ignores_column(self, tokenizer):
        strategy = _strategy(tokenizer, field_chat_template_kwargs=None)
        assert "Reasoning effort: 32" in _render(
            strategy, tokenizer, _row({"reasoning_effort": 128})
        )

    def test_null_valued_keys_fall_back(self, tokenizer):
        # Arrow struct columns fill keys a row lacks with null.
        strategy = _strategy(tokenizer, global_kwargs={"reasoning_effort": 64})
        text = _render(
            strategy,
            tokenizer,
            _row({"reasoning_effort": None, "enable_thinking": False}),
        )
        assert "Reasoning effort: 64" in text
        assert "Thinking: False" in text

    def test_with_tools(self, tokenizer):
        strategy = _strategy(tokenizer)
        text = _render(
            strategy, tokenizer, _row({"reasoning_effort": 256}, tools=TOOLS)
        )
        assert "Reasoning effort: 256" in text
        assert "Tools: lookup" in text

    def test_legacy_path(self, tokenizer):
        strategy = _strategy(tokenizer, roles_to_train=[], train_on_eos=None)
        tokenized = strategy.tokenize_prompt(_row({"reasoning_effort": 512}))
        assert "Reasoning effort: 512" in tokenizer.decode(tokenized["input_ids"])
        assert _trained_text(tokenizer, tokenized) == "That makes 6.<|im_end|>\n"

    def test_precedence_row_over_dataset_over_global(self, tokenizer):
        strategy = _strategy(
            tokenizer,
            global_kwargs={"reasoning_effort": 8, "enable_thinking": True},
            dataset_kwargs={"reasoning_effort": 64},
        )
        text = _render(strategy, tokenizer, _row())
        assert "Reasoning effort: 64" in text
        assert "Thinking: True" in text

        text = _render(
            strategy,
            tokenizer,
            _row({"reasoning_effort": 128, "enable_thinking": False}),
        )
        assert "Reasoning effort: 128" in text
        assert "Thinking: False" in text

        # The row override must not leak into the next row.
        assert "Reasoning effort: 64" in _render(strategy, tokenizer, _row())

        global_only = _strategy(tokenizer, global_kwargs={"reasoning_effort": 8})
        assert "Reasoning effort: 8" in _render(global_only, tokenizer, _row())


class TestRowKwargsLabels:
    @pytest.mark.parametrize("fallback", [False, True], ids=["text", "token-diff"])
    def test_masks_match_no_kwargs_case(self, tokenizer, fallback, monkeypatch):
        strategy = _strategy(tokenizer)
        if fallback:
            monkeypatch.setattr(strategy, "_build_turn_locator", lambda *a, **k: None)

        baseline = strategy.tokenize_prompt(_row(tools=TOOLS))
        tokenized = strategy.tokenize_prompt(
            _row({"reasoning_effort": 1024}, tools=TOOLS)
        )

        assert "Reasoning effort: 1024" in tokenizer.decode(tokenized["input_ids"])
        assert _trained_text(tokenizer, tokenized) == _trained_text(tokenizer, baseline)
        assert (
            _trained_text(tokenizer, tokenized)
            == "It is 4.<|im_end|>That makes 6.<|im_end|>"
        )

        # Only the header differs, so everything after it lines up token-for-token.
        shift = len(tokenized["input_ids"]) - len(baseline["input_ids"])
        assert shift > 0
        assert tokenized["input_ids"][shift:][-40:] == baseline["input_ids"][-40:]
        mask = [label != IGNORE_TOKEN_ID for label in tokenized["labels"]]
        base_mask = [label != IGNORE_TOKEN_ID for label in baseline["labels"]]
        assert not any(mask[:shift])
        assert mask[shift:] == base_mask

    def test_text_and_token_paths_agree(self, tokenizer):
        strategy = _strategy(tokenizer)
        row = _row({"reasoning_effort": 1024, "enable_thinking": True}, tools=TOOLS)
        turns = strategy.get_conversation_thread(row)
        template_kwargs = strategy._get_chat_template_kwargs(row)
        input_ids = strategy.prompter.build_prompt(
            turns, tools=TOOLS, template_kwargs=template_kwargs
        )
        locator = strategy._build_turn_locator(
            turns, TOOLS, input_ids, template_kwargs=template_kwargs
        )
        assert locator is not None
        for turn_idx in range(len(turns)):
            assert strategy.find_turn(
                turns=turns,
                turn_idx=turn_idx,
                tools=TOOLS,
                locator=locator,
                template_kwargs=template_kwargs,
            ) == strategy.find_turn(
                turns=turns,
                turn_idx=turn_idx,
                tools=TOOLS,
                template_kwargs=template_kwargs,
            )


class TestRowKwargsTypes:
    def test_int_is_preserved(self, tokenizer):
        strategy = _strategy(tokenizer)
        tokenized = strategy.tokenize_prompt(_row({"reasoning_effort": 128}))
        assert "Reasoning effort: 128" in tokenizer.decode(tokenized["input_ids"])

    def test_json_string_keeps_json_types(self, tokenizer):
        strategy = _strategy(tokenizer)
        text = _render(
            strategy,
            tokenizer,
            _row(json.dumps({"reasoning_effort": 128, "enable_thinking": False})),
        )
        assert "Reasoning effort: 128" in text
        assert "Thinking: False" in text

    def test_string_int_reaches_template_unchanged(self, tokenizer):
        strategy = _strategy(tokenizer)
        with pytest.raises(Exception, match="reasoning_effort must be an int"):
            strategy.tokenize_prompt(_row({"reasoning_effort": "128"}))


class TestRowKwargsValidation:
    @pytest.mark.parametrize(
        "value, match",
        [
            ([1, 2], "must be a dict"),
            (128, "must be a dict"),
            ("[1, 2]", "must be a dict"),
            ("reasoning_effort=128", "not valid JSON"),
            ({1: "x"}, "keys must be strings"),
        ],
    )
    def test_invalid_row_values_raise(self, tokenizer, value, match):
        strategy = _strategy(tokenizer)
        with pytest.raises(ValueError, match=match):
            strategy.tokenize_prompt(_row(value))

    @pytest.mark.parametrize(
        "value",
        [
            {"tools": TOOLS},
            {"add_generation_prompt": True},
            {"real_last_index": 1},
            {"tokenize": False},
            {"chat_template": "{{ 'x' }}"},
        ],
    )
    def test_reserved_row_keys_raise(self, tokenizer, value):
        strategy = _strategy(tokenizer)
        with pytest.raises(ValueError, match="may not set"):
            strategy.tokenize_prompt(_row(value))

    def test_reserved_dataset_keys_raise(self, tokenizer):
        with pytest.raises(ValueError, match="may not set"):
            _strategy(tokenizer, dataset_kwargs={"add_generation_prompt": True})

        with pytest.raises(ValueError, match="may not set"):
            SFTDataset(
                path="x",
                type="chat_template",
                chat_template_kwargs={"tools": []},
            )

    def test_schema_accepts_options(self):
        ds = SFTDataset(
            path="x",
            type="chat_template",
            field_chat_template_kwargs="template_kwargs",
            chat_template_kwargs={"reasoning_effort": 128},
        )
        assert ds.field_chat_template_kwargs == "template_kwargs"
        assert ds.chat_template_kwargs == {"reasoning_effort": 128}


class TestRowKwargsMultiprocess:
    ROWS = [
        {"reasoning_effort": 128},
        {},
        None,
        {"enable_thinking": False},
        {"reasoning_effort": 512, "enable_thinking": True},
        {"reasoning_effort": 8},
    ]

    def _check(self, tokenizer, tokenized, expected):
        assert len(tokenized) == len(expected)
        for row, (effort, thinking) in zip(tokenized, expected, strict=True):
            text = tokenizer.decode(row["input_ids"])
            assert f"Reasoning effort: {effort}" in text
            assert ("Thinking:" in text) == (thinking is not None)
            if thinking is not None:
                assert f"Thinking: {thinking}" in text
            assert _trained_text(tokenizer, row) == (
                "It is 4.<|im_end|>That makes 6.<|im_end|>"
            )

    def _expected(self, rows):
        return [
            (
                (row or {}).get("reasoning_effort", 32),
                (row or {}).get("enable_thinking"),
            )
            for row in rows
        ]

    def test_struct_column_num_proc(self, tokenizer):
        dataset = Dataset.from_list([_row(kw) for kw in self.ROWS])
        # Arrow unions the keys into one struct and fills the gaps with null.
        assert dataset[1]["template_kwargs"] == {
            "reasoning_effort": None,
            "enable_thinking": None,
        }
        strategy = _strategy(tokenizer)

        mapped = dataset.map(
            strategy.tokenize_prompt,
            batched=True,
            batch_size=2,
            num_proc=2,
            remove_columns=dataset.column_names,
        )
        self._check(tokenizer, mapped, self._expected(self.ROWS))

        queued = tokenize_with_work_queue(strategy, dataset, num_proc=2)
        self._check(tokenizer, queued, self._expected(self.ROWS))

    def test_json_string_column_mixed_types(self, tokenizer):
        rows = [
            {"reasoning_effort": 128},
            {"reasoning_effort": "high"},
            {},
            None,
        ]
        # Arrow can't hold an int and a str under one struct key.
        with pytest.raises(pa.ArrowInvalid):
            Dataset.from_list([_row(kw) for kw in rows[:2]])

        dataset = Dataset.from_list(
            [_row(json.dumps(kw) if kw is not None else None) for kw in rows]
        )
        mapped = dataset.map(
            _strategy(tokenizer).tokenize_prompt,
            batched=True,
            batch_size=2,
            num_proc=2,
            remove_columns=dataset.column_names,
        )
        efforts = ["128", "high", "32", "32"]
        for row, effort in zip(mapped, efforts, strict=True):
            assert f"Reasoning effort: {effort}" in tokenizer.decode(row["input_ids"])
