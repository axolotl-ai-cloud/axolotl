"""
Test dataset loading under various conditions.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizer

from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.data.rl import load_prepare_preference_datasets
from axolotl.utils.dict import DictDefault

from tests.constants import (
    ALPACA_MESSAGES_CONFIG_OG,
    ALPACA_MESSAGES_CONFIG_REVISION,
    SPECIAL_TOKENS,
)
from tests.hf_offline_utils import enable_hf_offline


class TestDatasetPreparation:
    """Test a configured dataloader."""

    @pytest.fixture
    def tokenizer(self, tokenizer_huggyllama) -> PreTrainedTokenizer:
        tokenizer_huggyllama.add_special_tokens(SPECIAL_TOKENS)
        yield tokenizer_huggyllama

    @pytest.fixture
    def dataset_fixture(self):
        yield Dataset.from_list(
            [
                {
                    "instruction": "Evaluate this sentence for spelling and grammar mistakes",
                    "input": "He finnished his meal and left the resturant",
                    "output": "He finished his meal and left the restaurant.",
                }
            ]
        )

    @pytest.mark.skip(reason="TODO: fix hf hub offline to work with HF rate limits")
    @enable_hf_offline
    def test_load_hub(self, tokenizer):
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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    @pytest.mark.skip("datasets bug with local datasets when offline")
    def test_load_local_hub(self, tokenizer):
        """Niche use case.  Verify that a local copy of a hub dataset can be loaded"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "mhenrichsen/alpaca_2k_test"
            tmp_ds_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_download(
                repo_id="mhenrichsen/alpaca_2k_test",
                repo_type="dataset",
                local_dir=tmp_ds_path,
            )
            # offline mode doesn't actually copy it to local_dir, so we
            # have to copy all the contents in the dir manually from the returned snapshot_path
            shutil.copytree(snapshot_path, tmp_ds_path, dirs_exist_ok=True)

            prepared_path = Path(tmp_dir) / "prepared"
            # Right now a local copy that doesn't fully conform to a dataset
            # must list data_files and ds_type otherwise the loader won't know
            # how to load it.
            cfg = DictDefault(
                {
                    "tokenizer_config": "HuggingFaceTB/SmolLM2-135M",
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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features
            shutil.rmtree(tmp_ds_path)

    @enable_hf_offline
    def test_load_from_save_to_disk(self, tokenizer, dataset_fixture):
        """Usual use case.  Verify datasets saved via `save_to_disk` can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_name = Path(tmp_dir) / "tmp_dataset"
            dataset_fixture.save_to_disk(str(tmp_ds_name))

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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    def test_load_from_dir_of_parquet(self, tokenizer, dataset_fixture):
        """Usual use case.  Verify a directory of parquet files can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_dir = Path(tmp_dir) / "tmp_dataset"
            tmp_ds_dir.mkdir()
            tmp_ds_path = tmp_ds_dir / "shard1.parquet"
            dataset_fixture.to_parquet(tmp_ds_path)

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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    def test_load_from_dir_of_json(self, tokenizer, dataset_fixture):
        """Standard use case.  Verify a directory of json files can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_dir = Path(tmp_dir) / "tmp_dataset"
            tmp_ds_dir.mkdir()
            tmp_ds_path = tmp_ds_dir / "shard1.json"
            dataset_fixture.to_json(tmp_ds_path)

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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    def test_load_from_single_parquet(self, tokenizer, dataset_fixture):
        """Standard use case.  Verify a single parquet file can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "tmp_dataset.parquet"
            dataset_fixture.to_parquet(tmp_ds_path)

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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    def test_load_from_single_json(self, tokenizer, dataset_fixture):
        """Standard use case.  Verify a single json file can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "tmp_dataset.json"
            dataset_fixture.to_json(tmp_ds_path)

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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @pytest.mark.skip(reason="TODO: fix hf offline mode for CI rate limits")
    @enable_hf_offline
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

        train_dataset, _ = load_prepare_preference_datasets(cfg)

        assert len(train_dataset) == 1800
        assert "conversation" in train_dataset.features

    @pytest.mark.skip(reason="TODO: fix hf hub offline to work with HF rate limits")
    @enable_hf_offline
    def test_load_hub_with_revision(self, tokenizer):
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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    def test_load_hub_with_revision_with_dpo(
        self, dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff
    ):
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

        # pylint: disable=duplicate-code
        with patch("axolotl.utils.data.rl.load_dataset_w_config") as mock_load_dataset:
            # Set up the mock to return different values on successive calls
            mock_load_dataset.return_value = (
                dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff
            )

            train_dataset, _ = load_prepare_preference_datasets(cfg)

            assert len(train_dataset) == 1800
            assert "conversation" in train_dataset.features

    @enable_hf_offline
    @pytest.mark.skip("datasets bug with local datasets when offline")
    def test_load_local_hub_with_revision(
        self, dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff, tokenizer
    ):
        """Verify that a local copy of a hub dataset can be loaded with a specific revision"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "mhenrichsen/alpaca_2k_test"
            tmp_ds_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_download(
                repo_id="mhenrichsen/alpaca_2k_test",
                repo_type="dataset",
                local_dir=tmp_ds_path,
                revision="d05c1cb",
            )
            shutil.copytree(snapshot_path, tmp_ds_path, dirs_exist_ok=True)

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

            with patch(
                "axolotl.utils.data.shared.load_dataset_w_config"
            ) as mock_load_dataset:
                # Set up the mock to return different values on successive calls
                mock_load_dataset.return_value = (
                    dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff
                )

                dataset, _ = load_tokenized_prepared_datasets(
                    tokenizer, cfg, prepared_path
                )

                assert len(dataset) == 2000
                assert "input_ids" in dataset.features
                assert "attention_mask" in dataset.features
                assert "labels" in dataset.features
                shutil.rmtree(tmp_ds_path)

    @enable_hf_offline
    def test_loading_local_dataset_folder(self, tokenizer):
        """Verify that a dataset downloaded to a local folder can be loaded"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_path = Path(tmp_dir) / "mhenrichsen/alpaca_2k_test"
            tmp_ds_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_download(
                repo_id="mhenrichsen/alpaca_2k_test",
                repo_type="dataset",
                local_dir=tmp_ds_path,
            )
            shutil.copytree(snapshot_path, tmp_ds_path, dirs_exist_ok=True)

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

            dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features
            shutil.rmtree(tmp_ds_path)
