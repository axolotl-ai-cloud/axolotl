"""Regression tests for the dataset wrapper config validation."""

import pytest

from axolotl.utils.data.wrappers import get_dataset_wrapper
from axolotl.utils.dict import DictDefault


def test_a_dataset_config_without_type_is_named():
    dataset_config = DictDefault({"path": "mhenrichsen/alpaca_2k_test"})
    cfg = DictDefault({"dataset_num_proc": 1, "skip_prepare_dataset": False})

    with pytest.raises(ValueError, match="dataset type is required"):
        get_dataset_wrapper(
            dataset_config=dataset_config,
            tokenizer=None,
            cfg=cfg,
            dataset_base_type=None,
            dataset=[],
        )


def test_skip_prepare_dataset_still_returns_the_dataset_untouched():
    dataset_config = DictDefault({"path": "some/dataset"})
    cfg = DictDefault({"dataset_num_proc": 1, "skip_prepare_dataset": True})
    dataset = [{"x": 1}]

    wrapped, prompter = get_dataset_wrapper(
        dataset_config=dataset_config,
        tokenizer=None,
        cfg=cfg,
        dataset_base_type=None,
        dataset=dataset,
    )

    assert wrapped is dataset
    assert prompter is None
