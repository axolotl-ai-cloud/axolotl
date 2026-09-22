"""Tests for dataset weight-based subsampling feature."""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pydantic import ValidationError

from axolotl.utils.config import validate_config
from axolotl.utils.data.shared import generate_dataset_hash_from_config, merge_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import SFTDataset


class TestSFTDatasetWeightSchema(unittest.TestCase):
    """Test the weight field on SFTDataset Pydantic schema."""

    def test_weight_none_by_default(self):
        ds = SFTDataset(path="test/dataset", type="alpaca")
        assert ds.weight is None

    def test_weight_valid_values(self):
        ds = SFTDataset(path="test/dataset", type="alpaca", weight=1.0)
        assert ds.weight == 1.0

        ds = SFTDataset(path="test/dataset", type="alpaca", weight=0.5)
        assert ds.weight == 0.5

        ds = SFTDataset(path="test/dataset", type="alpaca", weight=0.01)
        assert ds.weight == 0.01

    def test_weight_rejects_zero(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            SFTDataset(path="test/dataset", type="alpaca", weight=0.0)

    def test_weight_rejects_negative(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            SFTDataset(path="test/dataset", type="alpaca", weight=-0.5)

    def test_weight_rejects_above_one(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SFTDataset(path="test/dataset", type="alpaca", weight=1.5)


class TestWeightSurvivesValidation(unittest.TestCase):
    """Ensure the weight field is preserved through the config validation pipeline."""

    def _make_cfg(self, weight=None):
        ds = {"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}
        if weight is not None:
            ds["weight"] = weight
        return DictDefault(
            {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
                "learning_rate": 0.000001,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "datasets": [ds],
            }
        )

    def test_weight_preserved_after_validation(self):
        cfg = self._make_cfg(weight=0.5)
        checked = validate_config(cfg)
        assert checked.datasets[0].weight == 0.5

    def test_weight_preserved_with_capabilities(self):
        cfg = self._make_cfg(weight=0.3)
        checked = validate_config(
            cfg,
            capabilities={
                "bf16": "false",
                "n_gpu": 1,
                "compute_capability": "8.0",
            },
            env_capabilities={"torch_version": "2.6.0"},
        )
        assert checked.datasets[0].weight == 0.3

    def test_no_weight_means_none(self):
        cfg = self._make_cfg()
        checked = validate_config(cfg)
        assert checked.datasets[0].weight is None


class TestWeightSubsampling(unittest.TestCase):
    """Test the actual subsampling logic in _load_and_process_single_dataset."""

    def _make_dataset(self, n=100):
        return Dataset.from_dict(
            {
                "instruction": [f"instruction_{i}" for i in range(n)],
                "output": [f"output_{i}" for i in range(n)],
            }
        )

    @patch("axolotl.utils.data.sft.get_dataset_wrapper")
    @patch("axolotl.utils.data.sft.load_dataset_with_config")
    def test_weight_subsamples_dataset(self, mock_load, mock_wrapper):
        """When weight < 1.0, the dataset is shuffled and subsampled."""
        from axolotl.utils.data.sft import _load_and_process_single_dataset

        original_ds = self._make_dataset(100)
        mock_load.return_value = original_ds
        mock_wrapper.return_value = (original_ds, None)

        dataset_config = DictDefault(
            {"path": "test/ds", "type": "alpaca", "weight": 0.5}
        )
        cfg = DictDefault({"hf_use_auth_token": False, "seed": 42})

        # The wrapper is called with the subsampled dataset
        def capture_wrapper(**kwargs):
            ds = kwargs.get("dataset")
            return (ds, None)

        mock_wrapper.side_effect = capture_wrapper

        result_ds, _ = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=MagicMock(),
            split="train",
            seed=42,
        )

        assert len(result_ds) == 50

    @patch("axolotl.utils.data.sft.get_dataset_wrapper")
    @patch("axolotl.utils.data.sft.load_dataset_with_config")
    def test_weight_none_uses_full_dataset(self, mock_load, mock_wrapper):
        """When weight is None, the full dataset is used."""
        from axolotl.utils.data.sft import _load_and_process_single_dataset

        original_ds = self._make_dataset(100)
        mock_load.return_value = original_ds

        def capture_wrapper(**kwargs):
            return (kwargs.get("dataset"), None)

        mock_wrapper.side_effect = capture_wrapper

        dataset_config = DictDefault({"path": "test/ds", "type": "alpaca"})
        cfg = DictDefault({"hf_use_auth_token": False, "seed": 42})

        result_ds, _ = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=MagicMock(),
            split="train",
            seed=42,
        )

        assert len(result_ds) == 100

    @patch("axolotl.utils.data.sft.get_dataset_wrapper")
    @patch("axolotl.utils.data.sft.load_dataset_with_config")
    def test_weight_one_uses_full_dataset(self, mock_load, mock_wrapper):
        """When weight == 1.0, no subsampling occurs."""
        from axolotl.utils.data.sft import _load_and_process_single_dataset

        original_ds = self._make_dataset(100)
        mock_load.return_value = original_ds

        def capture_wrapper(**kwargs):
            return (kwargs.get("dataset"), None)

        mock_wrapper.side_effect = capture_wrapper

        dataset_config = DictDefault(
            {"path": "test/ds", "type": "alpaca", "weight": 1.0}
        )
        cfg = DictDefault({"hf_use_auth_token": False, "seed": 42})

        result_ds, _ = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=MagicMock(),
            split="train",
            seed=42,
        )

        assert len(result_ds) == 100

    @patch("axolotl.utils.data.sft.get_dataset_wrapper")
    @patch("axolotl.utils.data.sft.load_dataset_with_config")
    def test_weight_minimum_one_sample(self, mock_load, mock_wrapper):
        """Even with very small weights, at least 1 sample is kept."""
        from axolotl.utils.data.sft import _load_and_process_single_dataset

        original_ds = self._make_dataset(5)
        mock_load.return_value = original_ds

        def capture_wrapper(**kwargs):
            return (kwargs.get("dataset"), None)

        mock_wrapper.side_effect = capture_wrapper

        dataset_config = DictDefault(
            {"path": "test/ds", "type": "alpaca", "weight": 0.01}
        )
        cfg = DictDefault({"hf_use_auth_token": False, "seed": 42})

        result_ds, _ = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=MagicMock(),
            split="train",
            seed=42,
        )

        assert len(result_ds) >= 1

    @patch("axolotl.utils.data.sft.get_dataset_wrapper")
    @patch("axolotl.utils.data.sft.load_dataset_with_config")
    def test_weight_skipped_for_streaming(self, mock_load, mock_wrapper):
        """Weight-based subsampling is skipped for streaming datasets."""
        from axolotl.utils.data.sft import _load_and_process_single_dataset

        original_ds = self._make_dataset(100)
        mock_load.return_value = original_ds

        def capture_wrapper(**kwargs):
            return (kwargs.get("dataset"), None)

        mock_wrapper.side_effect = capture_wrapper

        dataset_config = DictDefault(
            {"path": "test/ds", "type": "alpaca", "weight": 0.5}
        )
        cfg = DictDefault({"hf_use_auth_token": False, "seed": 42})

        result_ds, _ = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=MagicMock(),
            split="train",
            seed=42,
            streaming=True,
        )

        # For streaming, len() would fail, but since we mock it returns a Dataset
        # The key test: weight is NOT applied in streaming mode
        assert len(result_ds) == 100

    @patch("axolotl.utils.data.sft.get_dataset_wrapper")
    @patch("axolotl.utils.data.sft.load_dataset_with_config")
    def test_weight_deterministic_with_seed(self, mock_load, mock_wrapper):
        """Same weight + seed produces the same subset."""
        from axolotl.utils.data.sft import _load_and_process_single_dataset

        original_ds = self._make_dataset(100)

        results = []
        for _ in range(2):
            mock_load.return_value = original_ds

            def capture_wrapper(**kwargs):
                return (kwargs.get("dataset"), None)

            mock_wrapper.side_effect = capture_wrapper

            dataset_config = DictDefault(
                {"path": "test/ds", "type": "alpaca", "weight": 0.3}
            )
            cfg = DictDefault({"hf_use_auth_token": False, "seed": 42})

            result_ds, _ = _load_and_process_single_dataset(
                dataset_config=dataset_config,
                cfg=cfg,
                tokenizer=MagicMock(),
                split="train",
                seed=42,
            )
            results.append(result_ds["instruction"])

        assert results[0] == results[1]


class TestDatasetHashIncludesWeight(unittest.TestCase):
    """Ensure dataset weight affects the cache hash."""

    def _make_cfg(self):
        return DictDefault(
            {
                "sequence_len": 1024,
                "sample_packing": False,
                "eval_sample_packing": False,
                "group_by_length": False,
                "kd_temperature": None,
            }
        )

    def test_different_weights_produce_different_hashes(self):
        cfg = self._make_cfg()
        ds_no_weight = [
            DictDefault(
                {
                    "path": "test/ds",
                    "type": "alpaca",
                    "shards": None,
                    "conversation": None,
                    "split": "train",
                    "temperature": None,
                    "weight": None,
                }
            )
        ]
        ds_with_weight = [
            DictDefault(
                {
                    "path": "test/ds",
                    "type": "alpaca",
                    "shards": None,
                    "conversation": None,
                    "split": "train",
                    "temperature": None,
                    "weight": 0.5,
                }
            )
        ]

        hash1 = generate_dataset_hash_from_config(cfg, ds_no_weight, "test-tokenizer")
        hash2 = generate_dataset_hash_from_config(cfg, ds_with_weight, "test-tokenizer")

        assert hash1 != hash2

    def test_same_weight_produces_same_hash(self):
        cfg = self._make_cfg()
        ds1 = [
            DictDefault(
                {
                    "path": "test/ds",
                    "type": "alpaca",
                    "shards": None,
                    "conversation": None,
                    "split": "train",
                    "temperature": None,
                    "weight": 0.5,
                }
            )
        ]
        ds2 = [
            DictDefault(
                {
                    "path": "test/ds",
                    "type": "alpaca",
                    "shards": None,
                    "conversation": None,
                    "split": "train",
                    "temperature": None,
                    "weight": 0.5,
                }
            )
        ]

        hash1 = generate_dataset_hash_from_config(cfg, ds1, "test-tokenizer")
        hash2 = generate_dataset_hash_from_config(cfg, ds2, "test-tokenizer")

        assert hash1 == hash2


class TestMergeWithWeightedDatasets(unittest.TestCase):
    """Integration test: verify weighted datasets merge correctly."""

    def test_weighted_datasets_merged_proportionally(self):
        ds1 = Dataset.from_dict({"text": [f"ds1_{i}" for i in range(100)]})
        ds2 = Dataset.from_dict({"text": [f"ds2_{i}" for i in range(100)]})

        # Simulate weight application: subsample ds2 to 50%
        ds2_subsampled = ds2.shuffle(seed=42).select(range(50))

        cfg = DictDefault(
            {
                "shuffle_merged_datasets": True,
                "shuffle_before_merging_datasets": False,
                "seed": 42,
                "curriculum_sampling": False,
            }
        )

        merged = merge_datasets([ds1, ds2_subsampled], cfg)
        assert len(merged) == 150  # 100 + 50


if __name__ == "__main__":
    unittest.main()
