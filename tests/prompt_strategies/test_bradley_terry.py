"""Tests for the bradley_terry.chat_template prompt strategy."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from axolotl.prompt_strategies.bradley_terry.chat_template import load
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    for word in "system user assistant be brief hello hi bye".split():
        vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


@pytest.fixture(name="strategy")
def fixture_strategy(tokenizer):
    cfg = DictDefault(
        {
            "chat_template": "chatml",
            "sequence_len": 64,
            "reward_model": True,
            "train_on_inputs": False,
        }
    )
    return load(tokenizer, cfg, {"chat_template": "chatml"})


def test_system_field_is_optional(strategy, tokenizer):
    row = {"input": "hello", "chosen": "hi", "rejected": "bye"}
    res = strategy.tokenize_prompt(dict(row))

    assert res == strategy.tokenize_prompt({**row, "system": None})
    assert tokenizer.convert_tokens_to_ids("system") not in res["chosen_ids"]
    assert tokenizer.convert_tokens_to_ids("hi") in res["chosen_ids"]
    assert tokenizer.convert_tokens_to_ids("bye") in res["rejected_ids"]


def test_system_field_is_used_when_present(strategy, tokenizer):
    res = strategy.tokenize_prompt(
        {"system": "be brief", "input": "hello", "chosen": "hi", "rejected": "bye"}
    )

    brief_id = tokenizer.convert_tokens_to_ids("brief")
    assert brief_id in res["chosen_ids"]
    assert brief_id in res["rejected_ids"]
