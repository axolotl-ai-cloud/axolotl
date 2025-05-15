"""Module containing tests for trainer utility functions."""

import unittest
from functools import partial

from axolotl.utils.trainer import truncate_or_drop_long_seq


# Test cases for truncate_or_drop_long_seq
class TestTruncateOrDropLongSeq(unittest.TestCase):
    """
    Test suite for truncate_or_drop_long_seq function.
    """

    def setUp(self):
        # Example sequence length settings
        """
        Sets up default sequence length parameters for the test cases.
        """
        self.sequence_len = 10
        self.min_sequence_len = 3

    def test_drop_mode_single(self):
        """
        Verifies that 'drop' mode correctly filters single sequence examples based on length.
        
        Tests that sequences shorter than the minimum, longer than the maximum, or empty are dropped,
        while sequences within the valid length range are kept.
        """
        handler = partial(
            truncate_or_drop_long_seq,
            sequence_len=self.sequence_len,
            min_sequence_len=self.min_sequence_len,
            handling="drop",
        )

        # Too short
        sample_short = {"input_ids": [1, 2]}
        self.assertFalse(handler(sample_short))

        # Too long
        sample_long = {"input_ids": list(range(self.sequence_len + 1))}
        self.assertFalse(handler(sample_long))

        # Just right
        sample_ok = {"input_ids": list(range(self.min_sequence_len))}
        self.assertTrue(handler(sample_ok))

        # Empty
        sample_empty = {"input_ids": []}
        self.assertFalse(handler(sample_empty))

    def test_truncate_mode_single(self):
        """
        Tests that 'truncate_or_drop_long_seq' correctly truncates or preserves single examples in "truncate" mode.
        
        Verifies that sequences longer than the maximum length are truncated, while sequences that are too short, empty, or within the valid range remain unchanged.
        """
        handler = partial(
            truncate_or_drop_long_seq,
            sequence_len=self.sequence_len,
            min_sequence_len=self.min_sequence_len,
            handling="truncate",
        )

        # Too short (should still be dropped implicitly by filter/map logic upstream,
        # but the function itself might return the sample or False based on impl.)
        # Current impl returns the original sample for map if too short, assuming upstream filters.
        # Let's refine this test - the function *itself* returns the sample if too short when truncating.
        sample_short = {"input_ids": [1, 2], "labels": [1, 2]}
        result_short = handler(sample_short)
        self.assertEqual(result_short["input_ids"], [1, 2])  # Unchanged

        # Too long
        original_long = list(range(self.sequence_len + 5))
        sample_long = {"input_ids": list(original_long), "labels": list(original_long)}
        result_long = handler(sample_long)
        self.assertEqual(len(result_long["input_ids"]), self.sequence_len)
        self.assertEqual(result_long["input_ids"], list(range(self.sequence_len)))
        self.assertEqual(len(result_long["labels"]), self.sequence_len)
        self.assertEqual(result_long["labels"], list(range(self.sequence_len)))

        # Just right
        sample_ok = {
            "input_ids": list(range(self.min_sequence_len)),
            "labels": list(range(self.min_sequence_len)),
        }
        result_ok = handler(sample_ok)
        self.assertEqual(len(result_ok["input_ids"]), self.min_sequence_len)
        self.assertEqual(result_ok, sample_ok)  # Should be unchanged

        # Empty
        sample_empty = {"input_ids": [], "labels": []}
        result_empty = handler(sample_empty)
        self.assertEqual(result_empty, sample_empty)  # Unchanged

    def test_drop_mode_batched(self):
        """
        Tests that the "drop" handling mode correctly filters batched input sequences based on length constraints.
        
        Verifies that sequences shorter than the minimum length, longer than the maximum length, or empty are dropped (returns False), while sequences within the valid range are kept (returns True).
        """
        handler = partial(
            truncate_or_drop_long_seq,
            sequence_len=self.sequence_len,
            min_sequence_len=self.min_sequence_len,
            handling="drop",
        )
        sample = {
            "input_ids": [
                [1, 2],  # Too short
                list(range(self.sequence_len + 1)),  # Too long
                list(range(self.sequence_len)),  # OK (len = 10)
                list(range(self.min_sequence_len)),  # OK (len = 3)
                [],  # Empty
            ]
        }
        expected = [False, False, True, True, False]
        self.assertEqual(handler(sample), expected)

    def test_truncate_mode_batched(self):
        """
        Tests that batched examples are correctly truncated in "truncate" mode.
        
        Verifies that sequences in both "input_ids" and "labels" longer than the maximum
        allowed length are truncated, while sequences that are too short or empty remain
        unchanged.
        """
        handler = partial(
            truncate_or_drop_long_seq,
            sequence_len=self.sequence_len,
            min_sequence_len=self.min_sequence_len,
            handling="truncate",
        )
        sample = {
            "input_ids": [
                [1, 2],  # Too short
                list(range(self.sequence_len + 5)),  # Too long
                list(range(self.sequence_len)),  # OK
                list(range(self.min_sequence_len)),  # OK
                [],  # Empty
            ],
            "labels": [  # Add labels to test truncation
                [1, 2],
                list(range(self.sequence_len + 5)),
                list(range(self.sequence_len)),
                list(range(self.min_sequence_len)),
                [],
            ],
        }

        result = handler(sample)

        # Expected results after truncation (too short and empty remain unchanged by this function)
        expected_input_ids = [
            [1, 2],  # Unchanged (too short)
            list(range(self.sequence_len)),  # Truncated
            list(range(self.sequence_len)),  # Unchanged (OK)
            list(range(self.min_sequence_len)),  # Unchanged (OK)
            [],  # Unchanged (Empty)
        ]
        expected_labels = [
            [1, 2],  # Unchanged (too short)
            list(range(self.sequence_len)),  # Truncated
            list(range(self.sequence_len)),  # Unchanged (OK)
            list(range(self.min_sequence_len)),  # Unchanged (OK)
            [],  # Unchanged (Empty)
        ]

        self.assertEqual(result["input_ids"], expected_input_ids)
        self.assertEqual(result["labels"], expected_labels)


if __name__ == "__main__":
    unittest.main()
