"""
Test dataset loading under various conditions.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from constants import (
    ALPACA_MESSAGES_CONFIG_OG,
    ALPACA_MESSAGES_CONFIG_REVISION,
    SPECIAL_TOKENS,
)
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.data.rl import load_prepare_dpo_datasets
from axolotl.utils.dict import DictDefault


class TestDatasetPreparation(unittest.TestCase):
    """Test a configured dataloader."""

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        # Alpaca dataset.
        self.dataset = Dataset.from_list(
            [
                {
                    "instruction": "Evaluate this sentence for spelling and grammar mistakes",
                    "input": "He finnished his meal and left the resturant",
                    "output": "He finished his meal and left the restaurant.",
                }
            ]
        )

    def test_load_hub(self):
        """Core use case.  Verify that processing data from the hub works"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared_path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 1024,
                    "datasets": [
                        {
                            "path": "mhenrichsen/alpaca_2k_test",
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_local_hub(self):
        """Niche use case.  Verify that a local copy of a hub dataset can be loaded"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "mhenrichsen/alpaca_2k_test"
            tmp_ds_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="mhenrichsen/alpaca_2k_test",
                repo_type="dataset",
                local_dir=tmp_ds_path,
            )

            prepared_path = Path(tmp_dir) / "prepared"
            # Right now a local copy that doesn't fully conform to a dataset
            # must list data_files and ds_type otherwise the loader won't know
            # how to load it.
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 1024,
                    "datasets": [
                        {
                            "path": "mhenrichsen/alpaca_2k_test",
                            "ds_type": "parquet",
                            "type": "alpaca",
                            "data_files": [
                                f"{tmp_ds_path}/alpaca_2000.parquet",
                            ],
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features
            shutil.rmtree(tmp_ds_path)

    def test_load_from_save_to_disk(self):
        """Usual use case.  Verify datasets saved via `save_to_disk` can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_name = Path(tmp_dir) / "tmp_dataset"
            self.dataset.save_to_disk(str(tmp_ds_name))

            prepared_path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 256,
                    "datasets": [
                        {
                            "path": str(tmp_ds_name),
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_from_dir_of_parquet(self):
        """Usual use case.  Verify a directory of parquet files can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_dir = Path(tmp_dir) / "tmp_dataset"
            tmp_ds_dir.mkdir()
            tmp_ds_path = tmp_ds_dir / "shard1.parquet"
            self.dataset.to_parquet(tmp_ds_path)

            prepared_path: Path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 256,
                    "datasets": [
                        {
                            "path": str(tmp_ds_dir),
                            "ds_type": "parquet",
                            "name": "test_data",
                            "data_files": [
                                str(tmp_ds_path),
                            ],
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_from_dir_of_json(self):
        """Standard use case.  Verify a directory of json files can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_dir = Path(tmp_dir) / "tmp_dataset"
            tmp_ds_dir.mkdir()
            tmp_ds_path = tmp_ds_dir / "shard1.json"
            self.dataset.to_json(tmp_ds_path)

            prepared_path: Path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 256,
                    "datasets": [
                        {
                            "path": str(tmp_ds_dir),
                            "ds_type": "json",
                            "name": "test_data",
                            "data_files": [
                                str(tmp_ds_path),
                            ],
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_from_single_parquet(self):
        """Standard use case.  Verify a single parquet file can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "tmp_dataset.parquet"
            self.dataset.to_parquet(tmp_ds_path)

            prepared_path: Path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 256,
                    "datasets": [
                        {
                            "path": str(tmp_ds_path),
                            "name": "test_data",
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_from_single_json(self):
        """Standard use case.  Verify a single json file can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "tmp_dataset.json"
            self.dataset.to_json(tmp_ds_path)

            prepared_path: Path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 256,
                    "datasets": [
                        {
                            "path": str(tmp_ds_path),
                            "name": "test_data",
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_hub_with_dpo(self):
        """Verify that processing dpo data from the hub works"""

        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "sequence_len": 1024,
                "rl": "dpo",
                "chat_template": "llama3",
                "datasets": [ALPACA_MESSAGES_CONFIG_OG],
            }
        )

        train_dataset, _ = load_prepare_dpo_datasets(cfg)

        assert len(train_dataset) == 1800
        assert "conversation" in train_dataset.features

    def test_load_hub_with_revision(self):
        """Verify that processing data from the hub works with a specific revision"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared_path = Path(tmp_dir) / "prepared"

            # make sure prepared_path is empty
            shutil.rmtree(prepared_path, ignore_errors=True)

            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 1024,
                    "datasets": [
                        {
                            "path": "mhenrichsen/alpaca_2k_test",
                            "type": "alpaca",
                            "revision": "d05c1cb",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    def test_load_hub_with_revision_with_dpo(self):
        """Verify that processing dpo data from the hub works with a specific revision"""

        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "sequence_len": 1024,
                "rl": "dpo",
                "chat_template": "llama3",
                "datasets": [ALPACA_MESSAGES_CONFIG_REVISION],
            }
        )

        train_dataset, _ = load_prepare_dpo_datasets(cfg)

        assert len(train_dataset) == 1800
        assert "conversation" in train_dataset.features

    def test_load_local_hub_with_revision(self):
        """Verify that a local copy of a hub dataset can be loaded with a specific revision"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "mhenrichsen/alpaca_2k_test"
            tmp_ds_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="mhenrichsen/alpaca_2k_test",
                repo_type="dataset",
                local_dir=tmp_ds_path,
                revision="d05c1cb",
            )

            prepared_path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 1024,
                    "datasets": [
                        {
                            "path": "mhenrichsen/alpaca_2k_test",
                            "ds_type": "parquet",
                            "type": "alpaca",
                            "data_files": [
                                f"{tmp_ds_path}/alpaca_2000.parquet",
                            ],
                            "revision": "d05c1cb",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features
            shutil.rmtree(tmp_ds_path)

    def test_loading_local_dataset_folder(self):
        """Verify that a dataset downloaded to a local folder can be loaded"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "mhenrichsen/alpaca_2k_test"
            tmp_ds_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="mhenrichsen/alpaca_2k_test",
                repo_type="dataset",
                local_dir=tmp_ds_path,
            )

            prepared_path = Path(tmp_dir) / "prepared"
            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 1024,
                    "datasets": [
                        {
                            "path": str(tmp_ds_path),
                            "type": "alpaca",
                        },
                    ],
                }
            )

            dataset, _ = load_tokenized_prepared_datasets(
                self.tokenizer, cfg, prepared_path
            )

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features
            shutil.rmtree(tmp_ds_path)


if __name__ == "__main__":
    unittest.main()
