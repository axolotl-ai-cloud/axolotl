"""
Test suite for functions in the axolotl.utils.data.utils module, focusing on the deduplicate_and_log_datasets function.

Additionally, this test suite includes tests for functions that indirectly call deduplicate_and_log_datasets during the execution of the preprocess command.
"""
import unittest

from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.utils.data import prepare_dataset
from axolotl.utils.data.constants import ALPACA_MESSAGES_CONFIG_REVISION, SPECIAL_TOKENS
from axolotl.utils.data.rl import load_prepare_dpo_datasets
from axolotl.utils.data.utils import deduplicate_and_log_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_processor, load_tokenizer


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


class TestDeduplicateRLDataset(unittest.TestCase):
    """Test a configured dataloader with deduplication."""

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.cfg = DictDefault(
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

    def test_load_with_deduplication(self):
        """Verify that loading with deduplication removes duplicates."""

        # Load the dataset using the deduplication setting
        train_dataset, _ = load_prepare_dpo_datasets(self.cfg)

        # Verify that the dataset has been deduplicated
        assert len(train_dataset) == 1800, "Dataset was not properly deduplicated"

    def test_load_without_deduplication(self):
        """Verify that loading without deduplication retains duplicates."""
        self.cfg.dataset_exact_deduplication = False
        # Load the dataset without deduplication
        train_dataset, _ = load_prepare_dpo_datasets(self.cfg)

        # Verify that the dataset retains duplicates
        assert (
            len(train_dataset) == 1800 * 2
        ), "Dataset deduplication occurred when it should not have"


class TestDeduplicateNonRL(unittest.TestCase):
    """Test prepare_dataset function with different configurations."""

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.cfg_1 = DictDefault(
            {
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


if __name__ == "__main__":
    unittest.main()
