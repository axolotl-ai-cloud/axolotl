"""
Config validation for per-dataset chat template kwargs and the rendering mix.
"""

import pytest
from pydantic import ValidationError

from axolotl.utils.data.shared import DATASET_HASH_DATASET_EXTRA_FIELDS
from axolotl.utils.schemas.datasets import SFTDataset

MIX = [
    {"kwargs": {"enable_thinking": True}, "weight": 0.3},
    {"kwargs": {"enable_thinking": False}, "weight": 0.7},
]


def test_valid_mix_is_accepted():
    dataset = SFTDataset(path="foo", type="chat_template", chat_template_kwargs_mix=MIX)

    assert [entry.weight for entry in dataset.chat_template_kwargs_mix] == [0.3, 0.7]
    assert dataset.chat_template_kwargs_mix[0].template_kwargs == {
        "enable_thinking": True
    }


def test_weight_defaults_to_one():
    dataset = SFTDataset(
        path="foo",
        type="chat_template",
        chat_template_kwargs_mix=[{"kwargs": {"enable_thinking": True}}],
    )

    assert dataset.chat_template_kwargs_mix[0].weight == 1.0


def test_plain_per_dataset_kwargs_are_accepted():
    dataset = SFTDataset(
        path="foo",
        type="chat_template",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert dataset.chat_template_kwargs == {"enable_thinking": False}
    assert dataset.chat_template_kwargs_mix is None


@pytest.mark.parametrize(
    "mix, match",
    [
        ([], "at least one entry"),
        ([{"kwargs": {}, "weight": 0.0}], "sum to more than 0"),
        ([{"kwargs": {}, "weight": -0.5}], "greater than or equal to 0"),
        ([{"kwargs": "enable_thinking", "weight": 1.0}], "kwargs"),
        ([{"kwargs": {}, "weight": "heavy"}], "weight"),
    ],
)
def test_invalid_mixes_are_rejected(mix, match):
    with pytest.raises(ValidationError, match=match):
        SFTDataset(path="foo", type="chat_template", chat_template_kwargs_mix=mix)


def test_mix_seed_is_optional_and_typed():
    dataset = SFTDataset(
        path="foo",
        type="chat_template",
        chat_template_kwargs_mix=MIX,
        chat_template_kwargs_mix_seed=7,
    )

    assert dataset.chat_template_kwargs_mix_seed == 7
    assert SFTDataset(path="foo").chat_template_kwargs_mix_seed is None


def test_mix_participates_in_the_prepared_dataset_hash():
    """Changing the mix must invalidate a cached preprocessed dataset."""
    for field in (
        "chat_template_kwargs",
        "chat_template_kwargs_mix",
        "chat_template_kwargs_mix_seed",
    ):
        assert field in DATASET_HASH_DATASET_EXTRA_FIELDS
