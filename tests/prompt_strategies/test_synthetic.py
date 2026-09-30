"""Tests for the synthetic dataset generator."""

import unittest
from itertools import groupby
from unittest.mock import MagicMock

import pytest
from datasets import Dataset
from pydantic import ValidationError

from axolotl.prompt_strategies._synthetic import SyntheticDatasetStrategy, load
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import SyntheticDataset


@pytest.mark.parametrize("sequence_length", [1, 16])
def test_short_default_sequences(sequence_length):
    config = SyntheticDataset(sequence_length=sequence_length, length=2)
    strategies = [
        SyntheticDatasetStrategy(sequence_length=sequence_length, length=2),
        load(
            MagicMock(vocab_size=1000),
            DictDefault(sequence_len=sequence_length),
            config.model_dump(),
        ),
    ]
    for strategy in strategies:
        for row in strategy.wrap_dataset(None):
            assert len(row["input_ids"]) == sequence_length
            assert row["labels"] == row["input_ids"]
            assert row["attention_mask"] == [1] * sequence_length


def test_load_normalizes_min_turn_length():
    strategy = load(
        MagicMock(vocab_size=1000),
        DictDefault(sequence_len=64),
        {"min_turn_length": "32", "max_turns": 4, "length": 2},
    )
    assert strategy.min_turn_length == 32
    for row in strategy.wrap_dataset(None):
        assert len(row["labels"]) == 64
        assert -100 in row["labels"]


@pytest.mark.parametrize("use_schema", [False, True])
def test_default_single_turn_labels_all_tokens(use_schema):
    ds_cfg = SyntheticDataset(length=2).model_dump() if use_schema else {"length": 2}
    strategy = load(MagicMock(vocab_size=1000), DictDefault(sequence_len=64), ds_cfg)
    assert strategy.min_turns == strategy.max_turns == 1
    assert strategy.input_fraction == 0
    for row in strategy.wrap_dataset(None):
        assert row["labels"] == row["input_ids"]


@pytest.mark.parametrize("use_schema", [False, True])
@pytest.mark.parametrize("fraction", [None, 0, 0.2])
def test_multi_turn_input_fraction_default(use_schema, fraction):
    ds_cfg = {"min_turns": 2, "max_turns": 4, "length": 2}
    if fraction is not None:
        ds_cfg["input_fraction"] = fraction
    if use_schema:
        ds_cfg = SyntheticDataset(**ds_cfg).model_dump()
    strategy = load(MagicMock(vocab_size=1000), DictDefault(sequence_len=256), ds_cfg)
    assert strategy.input_fraction == (0.25 if fraction is None else fraction)
    for row in strategy.wrap_dataset(None):
        assert (-100 in row["labels"]) == (fraction != 0)


def test_zero_input_fraction_with_short_turns():
    strategy = SyntheticDatasetStrategy(
        sequence_length=3,
        length=2,
        min_turns=3,
        max_turns=4,
        input_fraction=0,
        min_turn_length=1,
    )
    for row in strategy.wrap_dataset(None):
        assert row["labels"] == row["input_ids"]
    with pytest.raises(ValidationError, match="sequence_length"):
        SyntheticDataset(
            sequence_length=2,
            min_turns=3,
            max_turns=4,
            input_fraction=0,
            min_turn_length=1,
        )


@pytest.mark.parametrize("input_fraction", [0, 0.25])
@pytest.mark.parametrize("sequence_length", [64, 127, 257])
def test_full_sequences_and_turn_budget(input_fraction, sequence_length):
    ds_cfg = SyntheticDataset(
        sequence_length=sequence_length,
        min_turn_length=32,
        min_turns=2,
        max_turns=12,
        length=200,
        input_fraction=input_fraction,
        seed=42,
    ).model_dump()
    strategy = load(MagicMock(vocab_size=1000), DictDefault(sequence_len=512), ds_cfg)
    result = strategy.wrap_dataset(None)
    for row in result:
        length = len(row["input_ids"])
        assert length == sequence_length
        assert len(row["labels"]) == length
        assert row["attention_mask"] == [1] * length
        if input_fraction == 0:
            assert row["labels"] == row["input_ids"]
            continue
        runs = [
            len(list(tokens))
            for _, tokens in groupby(label == -100 for label in row["labels"])
        ]
        assert 2 <= len(runs) // 2 <= min(12, length // 32)
        turn_lengths = [a + b for a, b in zip(runs[::2], runs[1::2], strict=True)]
        assert min(turn_lengths) >= 32
        assert max(turn_lengths) - min(turn_lengths) <= 1
        assert sum(turn_lengths) == length
    assert result.to_dict() == strategy.wrap_dataset(None).to_dict()


def test_fixed_length_caps_turns_to_budget():
    result = SyntheticDatasetStrategy(
        sequence_length=65, length=10, min_turns=2, max_turns=100
    ).wrap_dataset(None)
    for row in result:
        runs = [list(tokens) for _, tokens in groupby(x == -100 for x in row["labels"])]
        assert [len(run) for run in runs] == [8, 25, 8, 24]


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.75])
def test_multi_turn_masking(fraction):
    strategy = SyntheticDatasetStrategy(
        sequence_length=1003,
        length=100,
        seed=42,
        min_turns=2,
        max_turns=5,
        input_fraction=fraction,
    )
    result = strategy.wrap_dataset(None)
    turn_counts = set()
    for row in result:
        assert len(row["input_ids"]) == len(row["labels"]) == 1003
        assert row["attention_mask"] == [1] * 1003
        runs = [
            (masked, len(list(tokens)))
            for masked, tokens in groupby(label == -100 for label in row["labels"])
        ]
        assert 4 <= len(runs) <= 10
        assert len(runs) % 2 == 0
        turn_counts.add(len(runs) // 2)
        for input_run, output_run in zip(runs[::2], runs[1::2], strict=True):
            assert input_run[0] is True
            assert output_run[0] is False
            total = input_run[1] + output_run[1]
            assert abs(input_run[1] - total * fraction) <= 0.5
        assert all(
            label == -100 or label == token
            for label, token in zip(row["labels"], row["input_ids"], strict=True)
        )
    assert turn_counts == {2, 3, 4, 5}
    assert result.to_dict() == strategy.wrap_dataset(None).to_dict()


@pytest.mark.parametrize("fraction", [1e-20, 0.5, 0.999999])
def test_multi_turn_minimum_length(fraction):
    result = SyntheticDatasetStrategy(
        sequence_length=6,
        length=2,
        min_turns=3,
        max_turns=4,
        input_fraction=fraction,
        min_turn_length=1,
    ).wrap_dataset(None)
    for row in result:
        assert row["labels"][::2] == [-100] * 3
        assert row["labels"][1::2] == row["input_ids"][1::2]


@pytest.mark.parametrize(
    "options",
    [
        {"min_turns": 0},
        {"max_turns": 0},
        {"min_turns": 5, "max_turns": 2},
        {"min_turns": 2, "max_turns": 2},
        {"input_fraction": -1},
        {"input_fraction": 1},
        {"input_fraction": 2},
        {"input_fraction": float("inf")},
        {"input_fraction": float("nan")},
        {"sequence_length": 7, "max_turns": 4},
        {"sequence_length": 16, "input_fraction": 0.25},
        {"sequence_length": 16, "max_turns": 4, "input_fraction": 0},
        {"min_turn_length": 0},
        {"sequence_length": 63, "min_turns": 2, "max_turns": 4},
    ],
)
def test_invalid_multi_turn_config(options):
    with pytest.raises(ValidationError):
        SyntheticDataset(**options)
    with pytest.raises(ValidationError):
        SyntheticDatasetStrategy(**options)


def test_load_multi_turn_schema_defaults():
    ds_cfg = SyntheticDataset(min_turns=2, max_turns=3, input_fraction=0.2).model_dump()
    tokenizer = MagicMock(vocab_size=1000)
    strategy = load(tokenizer, DictDefault(sequence_len=128), ds_cfg)
    assert strategy.min_turns == 2
    assert strategy.max_turns == 3
    assert strategy.input_fraction == 0.2
    assert strategy.sequence_length == 128
    assert strategy.max_input_id == 1000
    with pytest.raises(ValidationError, match="sequence_length"):
        load(tokenizer, DictDefault(sequence_len=5), ds_cfg)


class TestSyntheticDatasetStrategy(unittest.TestCase):
    def test_generates_correct_shape(self):
        strategy = SyntheticDatasetStrategy(
            sequence_length=128,
            length=50,
            min_input_id=1,
            max_input_id=1000,
            seed=42,
        )
        dummy = Dataset.from_dict({"text": [""]})
        result = strategy.wrap_dataset(dummy)

        assert len(result) == 50
        assert len(result[0]["input_ids"]) == 128
        assert len(result[0]["attention_mask"]) == 128
        assert len(result[0]["labels"]) == 128

    def test_attention_mask_all_ones(self):
        strategy = SyntheticDatasetStrategy(sequence_length=64, length=10, seed=0)
        dummy = Dataset.from_dict({"text": [""]})
        result = strategy.wrap_dataset(dummy)

        for row in result:
            assert all(v == 1 for v in row["attention_mask"])

    def test_labels_equal_input_ids(self):
        strategy = SyntheticDatasetStrategy(
            sequence_length=64, length=10, seed=0, input_fraction=0
        )
        dummy = Dataset.from_dict({"text": [""]})
        result = strategy.wrap_dataset(dummy)

        for row in result:
            assert row["input_ids"] == row["labels"]

    def test_input_id_range(self):
        strategy = SyntheticDatasetStrategy(
            sequence_length=64,
            length=100,
            min_input_id=500,
            max_input_id=600,
            seed=42,
        )
        dummy = Dataset.from_dict({"text": [""]})
        result = strategy.wrap_dataset(dummy)

        for row in result:
            for token_id in row["input_ids"]:
                assert 500 <= token_id < 600

    def test_seed_reproducibility(self):
        kwargs = dict(
            sequence_length=64, length=20, min_input_id=1, max_input_id=1000, seed=123
        )
        dummy = Dataset.from_dict({"text": [""]})

        result1 = SyntheticDatasetStrategy(**kwargs).wrap_dataset(dummy)
        result2 = SyntheticDatasetStrategy(**kwargs).wrap_dataset(dummy)

        for r1, r2 in zip(result1, result2, strict=True):
            assert r1["input_ids"] == r2["input_ids"]

    def test_different_seeds_differ(self):
        common = dict(sequence_length=64, length=20, min_input_id=1, max_input_id=1000)
        dummy = Dataset.from_dict({"text": [""]})

        result1 = SyntheticDatasetStrategy(seed=1, **common).wrap_dataset(dummy)
        result2 = SyntheticDatasetStrategy(seed=2, **common).wrap_dataset(dummy)

        any_different = any(
            r1["input_ids"] != r2["input_ids"]
            for r1, r2 in zip(result1, result2, strict=True)
        )
        assert any_different

    def test_load_function_with_ds_cfg(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 32000
        cfg = DictDefault({"sequence_len": 512, "train_on_inputs": False})
        ds_cfg = {
            "sequence_length": 256,
            "length": 5,
            "min_input_id": 10,
            "max_input_id": 100,
            "seed": 0,
        }

        strategy = load(tokenizer, cfg, ds_cfg=ds_cfg)
        assert isinstance(strategy, SyntheticDatasetStrategy)
        assert strategy.sequence_length == 256
        assert strategy.length == 5
        assert strategy.min_input_id == 10
        assert strategy.max_input_id == 100

    def test_load_defaults_from_cfg(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 32000
        cfg = DictDefault({"sequence_len": 1024, "train_on_inputs": False})

        strategy = load(tokenizer, cfg, ds_cfg={})
        assert strategy.sequence_length == 1024
        assert strategy.max_input_id == 32000
        assert strategy.length == 1000

    def test_load_with_no_ds_cfg(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 50000
        cfg = DictDefault({"sequence_len": 2048, "train_on_inputs": False})

        strategy = load(tokenizer, cfg)
        assert strategy.sequence_length == 2048
        assert strategy.max_input_id == 50000


if __name__ == "__main__":
    unittest.main()
