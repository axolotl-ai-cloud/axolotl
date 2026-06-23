"""Regression tests: prompt-strategy ``_tokenize`` must not crash on an empty result.

A field that tokenizes to nothing (e.g. an empty ``generation``/sub-field with a
tokenizer that does not prepend BOS) used to raise ``IndexError`` in these overrides
because they indexed ``result["input_ids"][-1]`` without the empty-guard the base
``PromptTokenizingStrategy._tokenize`` already has.
"""

import pytest
from transformers import AutoTokenizer

from axolotl.prompt_strategies.metharme import MetharmePromptTokenizingStrategy
from axolotl.prompt_tokenizers import ReflectionPromptTokenizingStrategy


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    # gpt2 does not prepend BOS, so tokenizing "" yields an empty input_ids list.
    return AutoTokenizer.from_pretrained("gpt2")


def _build(strategy_cls, tokenizer):
    strategy = strategy_cls.__new__(strategy_cls)
    strategy.tokenizer = tokenizer
    strategy.sequence_len = 2048
    return strategy


def test_metharme_tokenize_empty_does_not_crash(gpt2_tokenizer):
    strategy = _build(MetharmePromptTokenizingStrategy, gpt2_tokenizer)
    result = strategy._tokenize("")  # raised IndexError before the fix
    assert list(result["input_ids"]) == []


def test_reflection_tokenize_empty_does_not_crash(gpt2_tokenizer):
    strategy = _build(ReflectionPromptTokenizingStrategy, gpt2_tokenizer)
    result = strategy._tokenize("")  # raised IndexError before the fix
    assert list(result["input_ids"]) == []
