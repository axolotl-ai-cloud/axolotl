"""
Test suite for functions in the axolotl.utils.data.utils module, focusing on the deduplicate_and_log_datasets function.

Additionally, this test suite includes tests for functions that indirectly call deduplicate_and_log_datasets during the execution of the preprocess command.
"""
# pylint: disable=duplicate-code
# import logging
import unittest

from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.utils.data import prepare_dataset
from axolotl.utils.data.rl import load_prepare_dpo_datasets
from axolotl.utils.data.utils import deduplicate_and_log_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_processor, load_tokenizer

# logging.basicConfig(level=logging.INFO)


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

        # Verify sizes remain consistent post-deduplication
        train_size = len(train_rows)
        eval_size = len(eval_rows)
        self.assertEqual(
            train_size,
            len(train_dataset),
            "Size mismatch in train_dataset after deduplication",
        )
        self.assertEqual(
            eval_size,
            len(eval_dataset),
            "Size mismatch in eval_dataset after deduplication",
        )

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

        # Verify size remains consistent post-deduplication
        eval_size = len(eval_rows)
        self.assertEqual(
            eval_size,
            len(eval_dataset),
            "Size mismatch in eval_dataset after deduplication",
        )

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

        # Verify size remains consistent post-deduplication
        train_size = len(train_rows)
        self.assertEqual(
            train_size,
            len(train_dataset),
            "Size mismatch in train_dataset after deduplication",
        )

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
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=dataset, eval_dataset=dataset
        )

        # Verify that deduplicated datasets match expected rows
        expected_rows = set(tuple(row.values()) for row in expected_dataset)
        train_rows = set(tuple(row.values()) for row in train_dataset)
        eval_rows = set(tuple(row.values()) for row in eval_dataset)
        self.assertEqual(train_rows, expected_rows)
        self.assertEqual(eval_rows, expected_rows)

        # Verify size remains consistent post-deduplication
        train_size = len(train_rows)
        eval_size = len(eval_rows)
        self.assertEqual(
            train_size,
            len(train_dataset),
            "Size mismatch in train_dataset after deduplication",
        )
        self.assertEqual(
            eval_size,
            len(eval_dataset),
            "Size mismatch in eval_dataset after deduplication",
        )

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
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            train_dataset=dataset, eval_dataset=dataset
        )

        # Verify that deduplicated datasets match expected rows
        expected_rows = set(tuple(row.values()) for row in expected_dataset)
        train_rows = set(tuple(row.values()) for row in train_dataset)
        eval_rows = set(tuple(row.values()) for row in eval_dataset)
        self.assertEqual(train_rows, expected_rows)
        self.assertEqual(eval_rows, expected_rows)

        # Verify size remains consistent post-deduplication
        train_size = len(train_rows)
        eval_size = len(eval_rows)
        self.assertEqual(
            train_size,
            len(train_dataset),
            "Size mismatch in train_dataset after deduplication",
        )
        self.assertEqual(
            eval_size,
            len(eval_dataset),
            "Size mismatch in eval_dataset after deduplication",
        )


class TestDeduplicateRLDataset(unittest.TestCase):
    """Test a configured dataloader with deduplication."""

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )
        self.cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "sequence_len": 1024,
                "rl": "dpo",
                "chat_template": "llama3",
                "exact_deduplication": True,
                "datasets": [
                    {
                        "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                        "type": "chat_template.default",
                        "chat_template": "llama3",
                        "revision": "ea82cff",
                        "field_messages": "conversation",
                        "field_chosen": "chosen",
                        "field_rejected": "rejected",
                        "message_field_role": "role",
                        "message_field_content": "content",
                        "roles": {
                            "system": ["system"],
                            "user": ["user"],
                            "assistant": ["assistant"],
                        },
                    },
                    {
                        "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                        "type": "chat_template.default",
                        "chat_template": "llama3",
                        "revision": "ea82cff",
                        "field_messages": "conversation",
                        "field_chosen": "chosen",
                        "field_rejected": "rejected",
                        "message_field_role": "role",
                        "message_field_content": "content",
                        "roles": {
                            "system": ["system"],
                            "user": ["user"],
                            "assistant": ["assistant"],
                        },
                    },
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
        self.cfg.exact_deduplication = False
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
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )
        self.cfg_1 = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "sequence_len": 1024,
                "exact_deduplication": True,
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
        self.cfg_1.exact_deduplication = True

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
            "Train dataset should have 1800 samples after deduplication.",
        )

    def test_prepare_dataset_with_deduplication_eval(self):
        """Verify that prepare_dataset function processes the dataset correctly with deduplication."""
        self.cfg_1.exact_deduplication = True
        self.cfg_1.val_set_size = 3998
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
            2000,
            "Eval dataset should have 2000 samples after deduplication.",
        )
        # note that it should be 2000 because the val set size removed two unique elements. So there are 3996 elements with duplicates and 2 unique.
        # consequently by removing the duplicates we get 1998 + 2 = 2000 elements.

    def test_prepare_dataset_without_deduplication(self):
        """Verify that prepare_dataset function processes the dataset correctly without deduplication."""
        self.cfg_1.exact_deduplication = False
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
