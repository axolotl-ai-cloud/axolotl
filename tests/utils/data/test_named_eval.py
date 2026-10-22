"""Named evaluation grouping and preprocessing regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict
from pydantic import ValidationError

from axolotl.utils.data.evaluation import load_eval_datasets
from axolotl.utils.data.sft import _load_and_prepare_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import DPODataset, SFTDataset


def config():
    return DictDefault(
        {
            "test_datasets": [
                {"path": "a", "eval_name": "math"},
                {"path": "b", "eval_name": "math"},
                {"path": "c", "eval_name": "code"},
                {"path": "d"},
            ],
        }
    )


def test_groups_include_unnamed_only_in_aggregate():
    cfg = config()
    result = load_eval_datasets(
        cfg, lambda c: Dataset.from_dict({"path": [ds.path for ds in c.test_datasets]})
    )
    assert list(result) == ["", "math", "code"]
    assert result[""]["path"] == ["a", "b", "c", "d"]
    assert result["math"]["path"] == ["a", "b"]
    assert result["code"]["path"] == ["c"]
    assert len(cfg.test_datasets) == 4


def test_unnamed_preserves_dataset():
    cfg = DictDefault({"test_datasets": [{"path": "a"}]})
    dataset = Dataset.from_dict({"x": [1, 2]})
    assert load_eval_datasets(cfg, lambda _: dataset) is dataset


def test_sft_groups_use_preparation_and_sharding():
    cfg = config()

    def load(tokenizer, group_cfg, **kwargs):
        assert kwargs["split"] == "test"
        assert all("eval_name" not in ds for ds in group_cfg.test_datasets)
        return Dataset.from_dict(
            {"path": [ds.path for ds in group_cfg.test_datasets]}
        ), []

    with (
        patch(
            "axolotl.utils.data.sft._load_tokenized_prepared_datasets", side_effect=load
        ),
        patch(
            "axolotl.utils.data.sft._apply_dataset_sharding",
            side_effect=lambda ds, cfg: ds,
        ) as shard,
    ):
        _, result, _ = _load_and_prepare_datasets(SimpleNamespace(), cfg, split="test")
    assert result["math"]["path"] == ["a", "b"]
    assert len(result[""]) == 4
    assert shard.call_count == 3
    assert cfg.test_datasets[0].eval_name == "math"


@pytest.mark.parametrize("cls", [SFTDataset, DPODataset])
@pytest.mark.parametrize("name", ["", "two words", "x/y", "_private", "42"])
def test_invalid_names(cls, name):
    with pytest.raises(ValidationError):
        cls(path="test", eval_name=name)


def test_hf_name_is_independent():
    dataset = SFTDataset(path="test", name="subset", eval_name="math_2")
    assert dataset.name == "subset"
    assert dataset.eval_name == "math_2"


def test_preprocessed_cache_retains_groups(tmp_path):
    cfg = config()
    cfg.update(
        {
            "dataset_prepared_path": str(tmp_path / "cache"),
            "sequence_len": 8,
            "dataset_processes": 1,
            "dataset_exact_deduplication": True,
            "seed": 42,
        }
    )
    for index, ds_config in enumerate(cfg.test_datasets):
        source = tmp_path / ds_config.path
        tokens = [index + 1] * 4
        Dataset.from_dict(
            {
                "input_ids": [tokens, tokens],
                "attention_mask": [[1] * 4] * 2,
                "labels": [tokens, tokens],
            }
        ).save_to_disk(str(source))
        ds_config.path = str(source)
    tokenizer = SimpleNamespace(name_or_path="local-test", pad_token_id=0)
    _, first, _ = _load_and_prepare_datasets(tokenizer, cfg, split="test")
    assert len(first[""]) == 4
    assert len(first["math"]) == 2
    assert len(first["code"]) == 1
    with patch(
        "axolotl.utils.data.sft._load_raw_datasets",
        side_effect=AssertionError("cache missed"),
    ):
        _, cached, _ = _load_and_prepare_datasets(tokenizer, cfg, split="test")
    assert {name: ds.to_dict() for name, ds in first.items()} == {
        name: ds.to_dict() for name, ds in cached.items()
    }


def test_preference_groups_deduplicate_against_training(tmp_path):
    from axolotl.utils.data.rl import prepare_preference_datasets

    cfg = config()
    cfg.dataset_prepared_path = str(tmp_path)
    cfg.dataset_exact_deduplication = True

    def load(group_cfg, tokenizer, split):
        paths = [ds.path for ds in group_cfg.test_datasets]
        values = ["train", "train"] if split == "train" else ["train", *paths, *paths]
        return Dataset.from_dict({"prompt": values})

    with patch("axolotl.utils.data.rl._load_or_create_dataset_split", side_effect=load):
        train, evaluation = prepare_preference_datasets(cfg, SimpleNamespace())
    assert isinstance(evaluation, DatasetDict)
    assert train["prompt"] == ["train"]
    assert evaluation[""]["prompt"] == ["a", "b", "c", "d"]
    assert evaluation["math"]["prompt"] == ["a", "b"]
    assert evaluation["code"]["prompt"] == ["c"]


def test_packing_validates_each_group(tmp_path):
    from axolotl.utils.data.sft import _prepare_standard_dataset

    cfg = config()
    cfg.dataset_prepared_path = str(tmp_path)
    cfg.sample_packing = True
    cfg.eval_sample_packing = True
    dataset = Dataset.from_dict({"input_ids": [[1, 2, 3]]})
    evaluation = {"": dataset, "math": dataset}
    with (
        patch(
            "axolotl.utils.data.sft._load_and_prepare_datasets",
            side_effect=[(dataset, None, []), (None, evaluation, [])],
        ),
        patch(
            "axolotl.core.trainers.diffusion_lm.sampling.filter_native_diffusion_dataset",
            side_effect=lambda cfg, ds, split: ds,
        ),
        patch("axolotl.utils.data.sft.calculate_total_num_steps", side_effect=[1, 0]),
        pytest.raises(ValueError, match="math is too small"),
    ):
        _prepare_standard_dataset(cfg, SimpleNamespace(), None)


def test_streaming_training_loads_named_eval():
    from axolotl.utils.data.sft import _prepare_streaming_dataset

    cfg = config()
    cfg.max_steps = 10
    train = object()
    evaluation = {"": Dataset.from_dict({"input_ids": [[1, 2, 3]]})}
    with patch(
        "axolotl.utils.data.sft._load_and_prepare_datasets",
        side_effect=[(train, None, []), (None, evaluation, [])],
    ) as load:
        result = _prepare_streaming_dataset(cfg, SimpleNamespace(), None)
    assert result == (train, evaluation, 10, [])
    assert load.call_args_list[1].kwargs["split"] == "test"
    assert not load.call_args_list[1].kwargs.get("streaming", False)
