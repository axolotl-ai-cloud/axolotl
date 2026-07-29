"""
Tests for the KD collators' padding contract.
"""

from unittest.mock import MagicMock

import pytest

from axolotl.integrations.kd.collator import DataCollatorForKD


def _collator(padding_side):
    tokenizer = MagicMock()
    tokenizer.deprecation_warnings = {}
    tokenizer.padding_side = padding_side
    return DataCollatorForKD(tokenizer)


def _features():
    return [
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "target_logprobs": [[-0.1, -0.2]] * 3,
            "target_token_ids": [[2, 3]] * 3,
            "target_mask": [[1, 1]] * 3,
        }
    ]


def test_left_padding_with_teacher_data_raises():
    """Teacher rows are right-padded, so left-padded batches would be shifted."""
    with pytest.raises(ValueError, match="right padding"):
        _collator("left")(_features())


def test_right_padding_is_accepted():
    collator = _collator("right")
    collator.tokenizer.pad.return_value = {"input_ids": [[1, 2, 3]]}

    batch = collator(_features())

    assert batch["target_logprobs"].shape == (1, 3, 2)
    assert batch["target_token_ids"].shape == (1, 3, 2)
    assert batch["target_mask"].shape == (1, 3, 2)
