"""
Unit tests for data utility functions
"""

import unittest

from datasets import Dataset

from axolotl.utils.data.utils import handle_long_seq_in_dataset
from axolotl.utils.dict import DictDefault


class TestHandleLongSeqInDataset(unittest.TestCase):
    """
    Test class for handle_long_seq_in_dataset function
    """

    def test_drop_strategy_removes_long_sequences(self):
        """Test that 'drop' strategy removes sequences longer than sequence_len"""
        # Create dataset with mixed length sequences
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3],  # length 3 - keep
                    [1, 2, 3, 4, 5],  # length 5 - keep
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # length 11 - drop
                    [1, 2],  # length 2 - keep
                ]
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "drop",
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Should have dropped the sequence with length 11
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]["input_ids"]), 3)
        self.assertEqual(len(result[1]["input_ids"]), 5)
        self.assertEqual(len(result[2]["input_ids"]), 2)

    def test_drop_strategy_is_default(self):
        """Test that 'drop' is the default strategy when not specified"""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # length 11 - should drop
                ]
            }
        )

        cfg = DictDefault(
            {
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Should have dropped the long sequence
        self.assertEqual(len(result), 1)

    def test_truncate_strategy_truncates_long_sequences(self):
        """Test that 'truncate' strategy truncates sequences to sequence_len"""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3],  # length 3 - keep as is
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                    ],  # length 12 - truncate to 10
                ]
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "truncate",
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Should have 2 samples
        self.assertEqual(len(result), 2)
        # First sample unchanged
        self.assertEqual(len(result[0]["input_ids"]), 3)
        # Second sample truncated to 10
        self.assertEqual(len(result[1]["input_ids"]), 10)
        self.assertEqual(result[1]["input_ids"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_truncate_strategy_truncates_all_auxiliary_fields(self):
        """Test that truncation applies to all auxiliary fields consistently"""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                "labels": [
                    [-100, -100, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                ],
                "position_ids": [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                ],
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "truncate",
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # All fields should be truncated to 10
        self.assertEqual(len(result[0]["input_ids"]), 10)
        self.assertEqual(len(result[0]["attention_mask"]), 10)
        self.assertEqual(len(result[0]["labels"]), 10)
        self.assertEqual(len(result[0]["position_ids"]), 10)

        # Verify content is correct
        self.assertEqual(result[0]["input_ids"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(result[0]["attention_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(result[0]["labels"], [-100, -100, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(result[0]["position_ids"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_raise_strategy_raises_on_long_sequences(self):
        """Test that 'raise' strategy raises ValueError when encountering long sequences"""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # length 11 - should raise
                ]
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "raise",
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        with self.assertRaises(ValueError):
            handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

    def test_min_sequence_len_filters_short_sequences(self):
        """Test that sequences shorter than min_sample_len are filtered out"""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1],  # length 1 - drop (< min_sample_len=3)
                    [1, 2],  # length 2 - drop
                    [1, 2, 3],  # length 3 - keep
                    [1, 2, 3, 4, 5],  # length 5 - keep
                ]
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "drop",
                "min_sample_len": 3,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Should only keep sequences with length >= 3
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]["input_ids"]), 3)
        self.assertEqual(len(result[1]["input_ids"]), 5)

    def test_dataset_without_input_ids_column(self):
        """Test that datasets without 'input_ids' column are returned unchanged"""
        dataset = Dataset.from_dict(
            {
                "chosen": [1, 2, 3],
                "rejected": [4, 5, 6],
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "drop",
                "min_sample_len": 2,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Dataset should be unchanged
        self.assertEqual(len(result), len(dataset))
        self.assertListEqual(list(result.column_names), ["chosen", "rejected"])

    def test_truncate_filters_short_before_truncating(self):
        """Test that truncate strategy filters short sequences before truncating long ones

        This is important for efficiency - we should not waste time truncating
        sequences that will be filtered out anyway.
        """
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1],  # length 1 - filter out first
                    [1, 2, 3],  # length 3 - keep, no truncation needed
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                    ],  # length 12 - keep and truncate
                ]
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "truncate",
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Should have filtered out the first (short) sequence
        self.assertEqual(len(result), 2)
        # Second sample unchanged
        self.assertEqual(len(result[0]["input_ids"]), 3)
        # Third sample truncated to 10
        self.assertEqual(len(result[1]["input_ids"]), 10)

    def test_case_insensitive_strategy(self):
        """Test that excess_length_strategy is case-insensitive"""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                ]
            }
        )

        cfg = DictDefault(
            {
                "excess_length_strategy": "TRUNCATE",  # uppercase
                "min_sample_len": 2,
                "dataset_num_proc": 1,
                "is_preprocess": False,
            }
        )

        result = handle_long_seq_in_dataset(dataset, sequence_len=10, cfg=cfg)

        # Should still truncate
        self.assertEqual(len(result[0]["input_ids"]), 10)


if __name__ == "__main__":
    unittest.main()
