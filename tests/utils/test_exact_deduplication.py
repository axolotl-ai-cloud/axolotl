"""
test module for the axolotl.utils.data module
"""
import unittest
import logging
from datasets import Dataset
from axolotl.utils.data.utils import deduplicate_and_log_datasets

# Set up logging
logging.basicConfig(level=logging.INFO)

class TestDeduplicateAndLogDatasets(unittest.TestCase):
    """
    test class for deduplication function in data utils
    """

    def setUp(self):
        # Sample data with duplicates
        self.data = {
            "column1": ["apple", "banana", "apple", "orange", "banana"],
            "column2": [1, 2, 1, 3, 2],
            "column3": ["red", "yellow", "red", "orange", "yellow"]
        }

        # Expected result after deduplication
        self.expected_data = {
            "column1": ["apple", "banana", "orange"],
            "column2": [1, 2, 3],
            "column3": ["red", "yellow", "orange"]
        }

        # Convert to Dataset format
        self.dataset = Dataset.from_dict(self.data)
        self.expected_dataset = Dataset.from_dict(self.expected_data)

    def test_deduplication(self):
        # Run deduplication
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=self.dataset, eval_dataset=self.dataset
        )

        # Convert datasets to sets of row tuples for unordered comparison
        train_rows = set(tuple(row.values()) for row in train_dataset)
        eval_rows = set(tuple(row.values()) for row in eval_dataset)
        expected_rows = set(tuple(row.values()) for row in self.expected_dataset)

        # Verify that deduplicated datasets match expected rows
        self.assertEqual(train_rows, expected_rows, "Mismatch in train_dataset")
        self.assertEqual(eval_rows, expected_rows, "Mismatch in eval_dataset")

    def test_datasets_are_none(self):
        # Test when both datasets are None
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=None, eval_dataset=None
        )
        self.assertIsNone(train_dataset, "Expected train_dataset to be None")
        self.assertIsNone(eval_dataset, "Expected eval_dataset to be None")

    def test_only_train_is_none(self):
        # Test when only train_dataset is None
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=None, eval_dataset=self.dataset
        )
        self.assertIsNone(train_dataset, "Expected train_dataset to be None")

        # Verify that deduplication on eval_dataset works as expected
        eval_rows = set(tuple(row.values()) for row in eval_dataset)
        expected_rows = set(tuple(row.values()) for row in self.expected_dataset)
        self.assertEqual(eval_rows, expected_rows, "Mismatch in eval_dataset")

    def test_only_eval_is_none(self):
        # Test when only eval_dataset is None
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=self.dataset, eval_dataset=None
        )
        self.assertIsNone(eval_dataset, "Expected eval_dataset to be None")

        # Verify that deduplication on train_dataset works as expected
        train_rows = set(tuple(row.values()) for row in train_dataset)
        expected_rows = set(tuple(row.values()) for row in self.expected_dataset)
        self.assertEqual(train_rows, expected_rows, "Mismatch in train_dataset")

    def test_exact_duplicates(self):
        # Test when datasets are exact duplicates
        duplicate_data = {
            "column1": ["apple", "apple", "apple"],
            "column2": [1, 1, 1],
            "column3": ["red", "red", "red"]
        }
        expected_data = {
            "column1": ["apple"],
            "column2": [1],
            "column3": ["red"]
        }

        # Convert to Dataset format
        dataset = Dataset.from_dict(duplicate_data)
        expected_dataset = Dataset.from_dict(expected_data)

        # Run deduplication
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=dataset, eval_dataset=dataset
        )

        # Verify that deduplicated datasets match expected rows
        expected_rows = set(tuple(row.values()) for row in expected_dataset)
        self.assertEqual(set(tuple(row.values()) for row in train_dataset), expected_rows)
        self.assertEqual(set(tuple(row.values()) for row in eval_dataset), expected_rows)

    def test_partial_duplicates(self):
        # Test when only part of the dataset is a duplicate
        partial_duplicate_data = {
            "column1": ["apple", "banana", "apple"],
            "column2": [1, 2, 1],
            "column3": ["red", "yellow", "red"]
        }
        expected_data = {
            "column1": ["apple", "banana"],
            "column2": [1, 2],
            "column3": ["red", "yellow"]
        }

        # Convert to Dataset format
        dataset = Dataset.from_dict(partial_duplicate_data)
        expected_dataset = Dataset.from_dict(expected_data)

        # Run deduplication
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=dataset, eval_dataset=dataset
        )

        # Verify that deduplicated datasets match expected rows
        expected_rows = set(tuple(row.values()) for row in expected_dataset)
        self.assertEqual(set(tuple(row.values()) for row in train_dataset), expected_rows)
        self.assertEqual(set(tuple(row.values()) for row in eval_dataset), expected_rows)


if __name__ == "__main__":
    unittest.main()
