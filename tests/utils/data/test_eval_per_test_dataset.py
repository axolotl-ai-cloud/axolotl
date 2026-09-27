"""Tests for evaluating each of the `test_datasets` separately."""

import json
from pathlib import Path

import pytest
from datasets import Dataset

from axolotl.utils.data.sft import prepare_datasets
from axolotl.utils.data.shared import get_test_dataset_names
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline


def _write_alpaca_jsonl(path: Path, prefix: str, num_rows: int) -> str:
    with open(path, "w", encoding="utf-8") as fout:
        for idx in range(num_rows):
            row = {
                "instruction": f"{prefix} instruction {idx}",
                "input": "",
                "output": f"{prefix} answer {idx}",
            }
            fout.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture(name="data_cfg")
def fixture_data_cfg(tmp_path):
    return DictDefault(
        {
            "tokenizer_config": "huggyllama/llama-7b",
            "sequence_len": 256,
            "datasets": [
                {
                    "path": _write_alpaca_jsonl(tmp_path / "train.jsonl", "train", 4),
                    "type": "alpaca",
                }
            ],
            "test_datasets": [
                {
                    "path": _write_alpaca_jsonl(
                        tmp_path / "alpaca_eval.jsonl", "alpaca", 3
                    ),
                    "type": "alpaca",
                    "split": "train",
                },
                {
                    "path": _write_alpaca_jsonl(
                        tmp_path / "gsm8k_eval.jsonl", "gsm8k", 2
                    ),
                    "type": "alpaca",
                    "split": "train",
                },
            ],
            "dataset_prepared_path": str(tmp_path / "prepared"),
            "dataset_num_proc": 1,
            "micro_batch_size": 1,
            "batch_size": 1,
            "num_epochs": 1,
            "val_set_size": 0,
        }
    )


class TestGetTestDatasetNames:
    """Names used as metric prefixes for each test dataset."""

    def test_names_are_readable_and_indexed(self):
        names = get_test_dataset_names(
            [
                {"path": "tatsu-lab/alpaca"},
                {"path": "openai/gsm8k", "name": "main"},
                {"path": "/data/eval/my.eval.jsonl"},
                {"path": "json", "data_files": ["s3://bucket/held out.jsonl"]},
                {"path": "org/dataset", "name": ["a", "b"]},
            ]
        )
        assert names == [
            "alpaca_0",
            "gsm8k_main_1",
            "my_eval_2",
            "held_out_3",
            "dataset_4",
        ]

    def test_duplicate_paths_stay_unique(self):
        names = get_test_dataset_names([{"path": "tatsu-lab/alpaca"}] * 3)
        assert names == ["alpaca_0", "alpaca_1", "alpaca_2"]


class TestPrepareDatasetsPerTestDataset:
    """`prepare_datasets` with and without `eval_per_test_dataset`."""

    @enable_hf_offline
    def test_eval_split_is_dict_per_test_dataset(self, data_cfg, tokenizer_huggyllama):
        data_cfg.eval_per_test_dataset = True

        train_dataset, eval_dataset, _, _ = prepare_datasets(
            data_cfg, tokenizer_huggyllama
        )

        assert len(train_dataset) == 4
        assert isinstance(eval_dataset, dict)
        assert list(eval_dataset) == ["alpaca_eval_0", "gsm8k_eval_1"]
        assert {name: len(ds) for name, ds in eval_dataset.items()} == {
            "alpaca_eval_0": 3,
            "gsm8k_eval_1": 2,
        }
        for eval_split in eval_dataset.values():
            assert isinstance(eval_split, Dataset)
            assert "input_ids" in eval_split.features
            assert "labels" in eval_split.features

        # each test dataset is cached on its own, next to the train dataset
        prepared = Path(data_cfg.dataset_prepared_path)
        assert len([p for p in prepared.iterdir() if p.is_dir()]) == 3

        # a second run loads the per-dataset splits back from the prepared cache
        _, cached_eval_dataset, _, _ = prepare_datasets(data_cfg, tokenizer_huggyllama)
        assert {name: len(ds) for name, ds in cached_eval_dataset.items()} == {
            "alpaca_eval_0": 3,
            "gsm8k_eval_1": 2,
        }
        for name, eval_split in eval_dataset.items():
            assert cached_eval_dataset[name]["input_ids"] == eval_split["input_ids"]

    @enable_hf_offline
    def test_eval_split_is_merged_by_default(self, data_cfg, tokenizer_huggyllama):
        _, eval_dataset, _, _ = prepare_datasets(data_cfg, tokenizer_huggyllama)

        assert isinstance(eval_dataset, Dataset)
        assert len(eval_dataset) == 5
