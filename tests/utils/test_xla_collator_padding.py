"""On XLA every batch must be exactly sequence_len wide so the graph compiles once."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from axolotl.core.builders.causal import HFCausalTrainerBuilder
from axolotl.utils.dict import DictDefault

SEQ_LEN = 32


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    vocab = {"<pad>": 0, "<unk>": 1, **{f"w{i}": i + 2 for i in range(50)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>"
    )


def _padding_kwargs(xla: bool, **cfg):
    fake_builder: Any = SimpleNamespace(cfg=DictDefault(sequence_len=SEQ_LEN, **cfg))
    with patch("axolotl.core.builders.causal.is_xla", return_value=xla):
        return HFCausalTrainerBuilder._collator_padding_kwargs(fake_builder)


@pytest.mark.parametrize("lengths", [[3, 7], [1], [SEQ_LEN, 5], [17, 2, 30]])
def test_xla_batches_are_sequence_len_wide(tokenizer, lengths):
    collator = DataCollatorForSeq2Seq(tokenizer, **_padding_kwargs(xla=True))
    features = [{"input_ids": [2] * n, "labels": [2] * n} for n in lengths]
    batch = collator(features)
    assert batch["input_ids"].shape == (len(lengths), SEQ_LEN)
    assert batch["labels"].shape == (len(lengths), SEQ_LEN)
    assert batch["attention_mask"].shape == (len(lengths), SEQ_LEN)


def test_non_xla_padding_unchanged():
    assert _padding_kwargs(xla=False, pad_to_sequence_len=None) == {
        "padding": True,
        "pad_to_multiple_of": 64,
    }
