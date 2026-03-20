"""Tests for the synthetic dataset generator."""

import unittest
from unittest.mock import MagicMock

from datasets import Dataset

from axolotl.prompt_strategies._synthetic import SyntheticDatasetStrategy, load
from axolotl.utils.dict import DictDefault


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
        strategy = SyntheticDatasetStrategy(sequence_length=64, length=10, seed=0)
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
