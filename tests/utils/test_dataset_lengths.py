"""Tests for get_dataset_lengths column fallbacks and error reporting."""

import numpy as np
import pytest
from datasets import Dataset

from axolotl.utils.samplers.utils import get_dataset_lengths


def test_lengths_from_length_column():
    ds = Dataset.from_dict({"length": [3, 5], "input_ids": [[1, 2, 3], [1] * 5]})
    assert np.array_equal(get_dataset_lengths(ds), [3, 5])


def test_lengths_from_input_ids():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [1] * 5]})
    assert np.array_equal(get_dataset_lengths(ds), [3, 5])


def test_unprepared_dataset_raises_informative_error():
    """Raw (unprepared) rows must produce a diagnosis, not a bare KeyError."""
    ds = Dataset.from_dict({"messages": [["hi"], ["there"]]})
    with pytest.raises(ValueError, match="unprepared"):
        get_dataset_lengths(ds)
