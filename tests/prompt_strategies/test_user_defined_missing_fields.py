"""user_defined rows missing the configured fields must fail loudly and clearly."""

import pytest

from axolotl.prompt_strategies.user_defined import (
    UserDefinedDatasetConfig,
    load as user_defined_load,
)
from axolotl.utils.dict import DictDefault


def _make_strat(tokenizer):
    return user_defined_load(
        tokenizer,
        DictDefault({"train_on_inputs": False, "sequence_len": 2048}),
        UserDefinedDatasetConfig(
            field_instruction="question",
            field_input="context",
            field_output="answer",
        ),
    )


def test_missing_instruction_field_raises_clear_error(tokenizer_huggyllama):
    strat = _make_strat(tokenizer_huggyllama)
    row = {"context": "ctx", "answer": "Hi!"}
    with pytest.raises(ValueError, match="question"):
        strat.tokenize_prompt(row)


def test_missing_output_field_raises_clear_error(tokenizer_huggyllama):
    strat = _make_strat(tokenizer_huggyllama)
    row = {"question": "hello?", "context": "ctx"}
    with pytest.raises(ValueError, match="answer"):
        strat.tokenize_prompt(row)
