"""
Test suite for functions in the axolotl.utils.data.utils module, focusing on the deduplicate_and_log_datasets function.

Additionally, this test suite includes tests for functions that indirectly call deduplicate_and_log_datasets during the execution of the preprocess command.
"""

import hashlib
import unittest
from unittest.mock import patch

import pytest
from datasets import Dataset

from axolotl.utils.config import normalize_config
from axolotl.utils.data import prepare_dataset
from axolotl.utils.data.rl import load_prepare_preference_datasets
from axolotl.utils.data.utils import deduplicate_and_log_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_processor, load_tokenizer

from tests.constants import ALPACA_MESSAGES_CONFIG_REVISION
from tests.hf_offline_utils import enable_hf_offline


def verify_deduplication(actual_dataset, expected_dataset, dataset_name):
    """
    Validates deduplication results and size consistency.

    Parameters:
    - actual_dataset: Deduplicated dataset.
    - expected_dataset: Expected dataset.
    - dataset_name: Name of the dataset (e.g., 'train' or 'eval').

    Asserts:
    - Datasets match in content.
    - Dataset size matches unique row count.
    """
    # Convert datasets to sets of tuples for unordered comparison
    actual_rows = set(tuple(row.values()) for row in actual_dataset)
    expected_rows = set(tuple(row.values()) for row in expected_dataset)

    # Verify deduplication correctness
    assert actual_rows == expected_rows, f"Mismatch in {dataset_name} dataset"

    # Verify size consistency
    assert len(actual_rows) == len(
        actual_dataset
    ), f"Size mismatch in {dataset_name} dataset after deduplication"


class TestDeduplicateIndividualFunctions(unittest.TestCase):
    """
    test class for deduplication function in data utils
    """

    def setUp(self):
        # Sample data with duplicates
        self.data = {
            "column1": ["apple", "banana", "apple", "orange", "banana"],
            "column2": [1, 2, 1, 3, 2],
            "column3": ["red", "yellow", "red", "orange", "yellow"],
        }

        # Expected result after deduplication
        self.expected_data = {
            "column1": ["apple", "banana", "orange"],
            "column2": [1, 2, 3],
            "column3": ["red", "yellow", "orange"],
        }

        # Convert to Dataset format
        self.dataset = Dataset.from_dict(self.data)
        self.expected_dataset = Dataset.from_dict(self.expected_data)

    def test_deduplication(self):
        train_dataset, _, _ = deduplicate_and_log_datasets(train_dataset=self.dataset)
        _, eval_dataset, _ = deduplicate_and_log_datasets(eval_dataset=self.dataset)

        verify_deduplication(train_dataset, self.expected_dataset, "train_dataset")
        verify_deduplication(eval_dataset, self.expected_dataset, "eval_dataset")

    def test_datasets_are_none(self):
        # Test when both datasets are None
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=None, eval_dataset=None
        )
        self.assertIsNone(train_dataset, "Expected train_dataset to be None")
        self.assertIsNone(eval_dataset, "Expected eval_dataset to be None")

    def test_only_train_is_none(self):
        # Test when only train_dataset is None
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=None, eval_dataset=self.dataset
        )
        self.assertIsNone(train_dataset, "Expected train_dataset to be None")
        verify_deduplication(eval_dataset, self.expected_dataset, "eval_dataset")

    def test_only_eval_is_none(self):
        # Test when only eval_dataset is None
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=self.dataset, eval_dataset=None
        )
        self.assertIsNone(eval_dataset, "Expected eval_dataset to be None")
        verify_deduplication(train_dataset, self.expected_dataset, "train_dataset")

    def test_exact_duplicates(self):
        # Test when datasets are exact duplicates
        duplicate_data = {
            "column1": ["apple", "apple", "apple"],
            "column2": [1, 1, 1],
            "column3": ["red", "red", "red"],
        }
        expected_data = {"column1": ["apple"], "column2": [1], "column3": ["red"]}

        # Convert to Dataset format
        dataset = Dataset.from_dict(duplicate_data)
        expected_dataset = Dataset.from_dict(expected_data)

        # Run deduplication
        train_dataset, _, _ = deduplicate_and_log_datasets(train_dataset=dataset)
        _, eval_dataset, _ = deduplicate_and_log_datasets(eval_dataset=dataset)

        verify_deduplication(train_dataset, expected_dataset, "train_dataset")
        verify_deduplication(eval_dataset, expected_dataset, "eval_dataset")

    def test_partial_duplicates(self):
        # Test when only part of the dataset is a duplicate
        partial_duplicate_data = {
            "column1": ["apple", "banana", "apple"],
            "column2": [1, 2, 1],
            "column3": ["red", "yellow", "red"],
        }
        expected_data = {
            "column1": ["apple", "banana"],
            "column2": [1, 2],
            "column3": ["red", "yellow"],
        }

        # Convert to Dataset format
        dataset = Dataset.from_dict(partial_duplicate_data)
        expected_dataset = Dataset.from_dict(expected_data)

        # Run deduplication
        train_dataset, _, _ = deduplicate_and_log_datasets(train_dataset=dataset)
        _, eval_dataset, _ = deduplicate_and_log_datasets(eval_dataset=dataset)

        verify_deduplication(train_dataset, expected_dataset, "train_dataset")
        verify_deduplication(eval_dataset, expected_dataset, "eval_dataset")

    def test_combined_duplicates_empty(self):
        # Test when only part of the dataset is a duplicate
        partial_duplicate_data = {
            "column1": ["apple", "banana", "apple"],
            "column2": [1, 2, 1],
            "column3": ["red", "yellow", "red"],
        }
        expected_data_train = {
            "column1": ["apple", "banana"],
            "column2": [1, 2],
            "column3": ["red", "yellow"],
        }
        expected_data_eval = {
            "column1": [],
            "column2": [],
            "column3": [],
        }

        # Convert to Dataset format
        dataset = Dataset.from_dict(partial_duplicate_data)
        expected_dataset_train = Dataset.from_dict(expected_data_train)
        expected_dataset_eval = Dataset.from_dict(expected_data_eval)

        # Run deduplication
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=dataset, eval_dataset=dataset
        )

        verify_deduplication(train_dataset, expected_dataset_train, "train_dataset")
        verify_deduplication(eval_dataset, expected_dataset_eval, "eval_dataset")

    def test_combined_duplicates_one(self):
        # Test when only part of the dataset is a duplicate
        partial_duplicate_data_train = {
            "column1": ["apple", "banana", "apple"],
            "column2": [1, 2, 1],
            "column3": ["red", "yellow", "red"],
        }
        partial_duplicate_data_eval = {
            "column1": ["apple", "orange", "apple"],
            "column2": [1, 2, 1],
            "column3": ["red", "orange", "red"],
        }
        expected_data_train = {
            "column1": ["apple", "banana"],
            "column2": [1, 2],
            "column3": ["red", "yellow"],
        }
        expected_data_eval = {
            "column1": ["orange"],
            "column2": [2],
            "column3": ["orange"],
        }

        # Convert to Dataset format
        dataset_train = Dataset.from_dict(partial_duplicate_data_train)
        dataset_eval = Dataset.from_dict(partial_duplicate_data_eval)
        expected_dataset_train = Dataset.from_dict(expected_data_train)
        expected_dataset_eval = Dataset.from_dict(expected_data_eval)

        # Run deduplication
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=dataset_train, eval_dataset=dataset_eval
        )

        verify_deduplication(train_dataset, expected_dataset_train, "train_dataset")
        verify_deduplication(eval_dataset, expected_dataset_eval, "eval_dataset")


class TestDeduplicateRLDataset:
    """Test a configured dataloader with deduplication."""

    @pytest.fixture
    def cfg(self):
        fixture = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "sequence_len": 1024,
                "rl": "dpo",
                "chat_template": "llama3",
                "dataset_exact_deduplication": True,
                "datasets": [
                    ALPACA_MESSAGES_CONFIG_REVISION,
                    ALPACA_MESSAGES_CONFIG_REVISION,
                ],
            }
        )
        yield fixture

    @enable_hf_offline
    def test_load_with_deduplication(
        self,
        cfg,
        dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff,
        tokenizer_huggyllama,
    ):
        """Verify that loading with deduplication removes duplicates."""

        # pylint: disable=duplicate-code
        with (
            patch("axolotl.utils.data.rl.load_dataset_w_config") as mock_load_dataset,
            patch("axolotl.utils.models.load_tokenizer") as mock_load_tokenizer,
        ):
            # Set up the mock to return different values on successive calls
            mock_load_dataset.side_effect = [
                dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff,
                dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff,
            ]
            mock_load_tokenizer.return_value = tokenizer_huggyllama

            train_dataset, _ = load_prepare_preference_datasets(cfg)

            # Verify that the dataset has been deduplicated
            assert len(train_dataset) == 1800, "Dataset was not properly deduplicated"

    @enable_hf_offline
    def test_load_without_deduplication(
        self,
        cfg,
        dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff,
        tokenizer_huggyllama,
    ):
        # pylint: disable=duplicate-code
        with (
            patch("axolotl.utils.data.rl.load_dataset_w_config") as mock_load_dataset,
            patch("axolotl.utils.models.load_tokenizer") as mock_load_tokenizer,
        ):
            # Set up the mock to return different values on successive calls
            mock_load_dataset.side_effect = [
                dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff,
                dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff,
            ]
            mock_load_tokenizer.return_value = tokenizer_huggyllama

            cfg.dataset_exact_deduplication = False
            # Load the dataset without deduplication
            train_dataset, _ = load_prepare_preference_datasets(cfg)

            # Verify that the dataset retains duplicates
            assert (
                len(train_dataset) == 1800 * 2
            ), "Dataset deduplication occurred when it should not have"


class TestDeduplicateNonRL(unittest.TestCase):
    """Test prepare_dataset function with different configurations."""

    @enable_hf_offline
    def setUp(self) -> None:
        self.cfg_1 = DictDefault(
            {
                "base_model": "huggyllama/llama-7b",
                "tokenizer_config": "huggyllama/llama-7b",
                "sequence_len": 1024,
                "dataset_exact_deduplication": True,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "val_set_size": 0.0,
                "gradient_accumulation_steps": 4,
                "batch_size": 10,
                "micro_batch_size": 10,
                "num_epochs": 1,
            }
        )
        normalize_config(self.cfg_1)

    @pytest.mark.skip(reason="TODO: fix hf hub offline to work with HF rate limits")
    @enable_hf_offline
    def test_prepare_dataset_with_deduplication_train(self):
        """Verify that prepare_dataset function processes the dataset correctly with deduplication."""
        self.cfg_1.dataset_exact_deduplication = True

        # Load tokenizer and processor
        tokenizer = load_tokenizer(self.cfg_1)
        processor = (
            load_processor(self.cfg_1, tokenizer=tokenizer)
            if self.cfg_1.processor_type
            else None
        )

        # Prepare dataset using the prepare_dataset function
        train_dataset, _, _, _ = prepare_dataset(
            self.cfg_1,
            tokenizer,
            processor=processor,
        )

        self.assertEqual(
            len(train_dataset),
            2000,
            "Train dataset should have 2000 samples after deduplication.",
        )

    @pytest.mark.skip(reason="TODO: fix hf hub offline to work with HF rate limits")
    @enable_hf_offline
    def test_prepare_dataset_with_deduplication_eval(self):
        """Verify that prepare_dataset function processes the dataset correctly with deduplication."""
        self.cfg_1.dataset_exact_deduplication = True
        self.cfg_1.val_set_size = 0.5
        # Load tokenizer and processor
        tokenizer = load_tokenizer(self.cfg_1)
        processor = (
            load_processor(self.cfg_1, tokenizer=tokenizer)
            if self.cfg_1.processor_type
            else None
        )

        # Prepare dataset using the prepare_dataset function
        _, eval_dataset, _, _ = prepare_dataset(
            self.cfg_1,
            tokenizer,
            processor=processor,
        )

        self.assertEqual(
            len(eval_dataset),
            1000,
            "Eval dataset should have 2000 samples after deduplication.",
        )

    @pytest.mark.skip(reason="TODO: fix hf hub offline to work with HF rate limits")
    @enable_hf_offline
    def test_prepare_dataset_without_deduplication(self):
        """Verify that prepare_dataset function processes the dataset correctly without deduplication."""
        self.cfg_1.dataset_exact_deduplication = False
        self.cfg_1.val_set_size = 0.1
        # Load tokenizer and processor
        tokenizer = load_tokenizer(self.cfg_1)
        processor = (
            load_processor(self.cfg_1, tokenizer=tokenizer)
            if self.cfg_1.processor_type
            else None
        )

        # Prepare dataset using the prepare_dataset function
        train_dataset, eval_dataset, _, _ = prepare_dataset(
            self.cfg_1,
            tokenizer,
            processor=processor,
        )

        # Verify that the dataset has been prepared correctly
        self.assertEqual(
            len(train_dataset),
            1800 * 2,
            "Train dataset should have 3600 samples without deduplication.",
        )
        self.assertEqual(
            len(eval_dataset),
            200 * 2,
            "Train dataset should have 400 samples after deduplication.",
        )


class TestWrongCollisions(unittest.TestCase):
    """Creating mock datasets for testing wrong collisions"""

    def setUp(self):
        self.train_data = {"text": ["sample 5", "sample 6"], "label": [1, 2]}
        self.eval_data = {
            "text": [
                "sample 5",
                "sample 7",
            ],  # Different label but same text as in train_data
            "label": [2, 3],
        }
        self.dataset_data = {
            "text": ["sample 5", "sample 9", "sample 5"],
            "label": [1, 2, 8],
        }
        self.train_dataset = Dataset.from_dict(self.train_data)
        self.eval_dataset = Dataset.from_dict(self.eval_data)
        self.dataset = Dataset.from_dict(self.dataset_data)

    @patch(
        "axolotl.utils.data.utils.sha256",
        side_effect=lambda x: (
            hashlib.sha256("forced_collision_hash".encode("utf-8")).hexdigest()
            if "sample 5" in x
            else hashlib.sha256(x.encode("utf-8")).hexdigest()
        ),
    )
    def test_deduplication_wrong_collision_train_eval(self, _mock_sha256):
        dedup_train, dedup_eval, _ = deduplicate_and_log_datasets(
            train_dataset=self.train_dataset, eval_dataset=self.eval_dataset
        )
        self.assertEqual(
            len(dedup_train),
            2,
            "train dataset should not deduplicate rows with forced hash collisions but different labels.",
        )
        self.assertEqual(
            len(dedup_eval),
            2,
            "Eval dataset should not deduplicate rows with forced hash collisions but different labels.",
        )
        self.assertEqual(
            len(dedup_eval),
            len(self.eval_dataset),
            "The output eval dataset should have the same number of rows as the input eval dataset.",
        )
        self.assertEqual(
            str(dedup_eval),
            str(self.eval_dataset),
            "The string representation of the output eval dataset should be identical to the input eval dataset.",
        )

    def test_deduplication_dataset_only(self):
        _, _, dedup_dataset = deduplicate_and_log_datasets(dataset=self.dataset)
        self.assertEqual(
            len(dedup_dataset), 3, "Dataset should have all original values"
        )
        self.assertEqual(
            str(dedup_dataset),
            str(self.dataset),
            "The string representation of the output dataset should not differ.",
        )


if __name__ == "__main__":
    unittest.main()
