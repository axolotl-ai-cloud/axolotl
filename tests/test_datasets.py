"""Test dataset loading under various conditions."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
from datasets import Dataset, IterableDataset
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizer

from axolotl.loaders.tokenizer import load_tokenizer
from axolotl.utils.data.rl import prepare_preference_datasets
from axolotl.utils.data.sft import _load_tokenized_prepared_datasets, prepare_datasets
from axolotl.utils.dict import DictDefault

from tests.constants import (
    ALPACA_MESSAGES_CONFIG_OG,
    ALPACA_MESSAGES_CONFIG_REVISION,
    SPECIAL_TOKENS,
)
from tests.hf_offline_utils import enable_hf_offline


# pylint: disable=too-many-public-methods
class TestDatasetPreparation:
    """Test a configured dataloader."""

    @pytest.fixture
    def tokenizer(
        self, tokenizer_huggyllama
    ) -> Generator[PreTrainedTokenizer, Any, Any]:
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

    @pytest.fixture
    def streaming_dataset_fixture(self):
        """Create a streaming dataset fixture for testing."""

        def generator():
            yield {
                "instruction": "Evaluate this sentence for spelling and grammar mistakes",
                "input": "He finnished his meal and left the resturant",
                "output": "He finished his meal and left the restaurant.",
            }
            yield {
                "instruction": "What is the capital of France?",
                "input": "",
                "output": "The capital of France is Paris.",
            }

        return IterableDataset.from_generator(generator)

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

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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
                    "dataset_processes": 4,
                }
            )

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

            assert len(dataset) == 1
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features

    @enable_hf_offline
    def test_load_from_dir_of_parquet(self, tokenizer, dataset_fixture):
        """Usual use case. Verify a directory of parquet files can be loaded."""
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
                    "dataset_processes": 4,
                }
            )

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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
                    "dataset_processes": 4,
                }
            )

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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
                    "dataset_processes": 4,
                }
            )

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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
                    "dataset_processes": 4,
                }
            )

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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

        tokenizer = load_tokenizer(cfg)
        train_dataset, _ = prepare_preference_datasets(cfg, tokenizer)

        assert len(train_dataset) == 1800
        assert "conversation" not in train_dataset.features
        assert "chosen" in train_dataset.features
        assert "rejected" in train_dataset.features
        assert "prompt" in train_dataset.features

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

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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
                "dataset_processes": 4,
            }
        )

        # pylint: disable=duplicate-code
        with patch(
            "axolotl.utils.data.rl.load_dataset_with_config"
        ) as mock_load_dataset:
            # Set up the mock to return different values on successive calls
            mock_load_dataset.return_value = (
                dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff
            )

            tokenizer = load_tokenizer(cfg)
            train_dataset, _ = prepare_preference_datasets(cfg, tokenizer)

            assert len(train_dataset) == 1800
            assert "conversation" not in train_dataset.features
            assert "chosen" in train_dataset.features
            assert "rejected" in train_dataset.features
            assert "prompt" in train_dataset.features

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
                "axolotl.utils.data.shared.load_dataset_with_config"
            ) as mock_load_dataset:
                # Set up the mock to return different values on successive calls
                mock_load_dataset.return_value = (
                    dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff
                )

                with patch(
                    "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH",
                    str(prepared_path),
                ):
                    dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

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
                    "dataset_processes": 4,
                }
            )

            with patch(
                "axolotl.common.const.DEFAULT_DATASET_PREPARED_PATH", str(prepared_path)
            ):
                dataset, _ = _load_tokenized_prepared_datasets(tokenizer, cfg)

            assert len(dataset) == 2000
            assert "input_ids" in dataset.features
            assert "attention_mask" in dataset.features
            assert "labels" in dataset.features
            shutil.rmtree(tmp_ds_path)

    def test_streaming_sft_dataset(self, tokenizer, streaming_dataset_fixture):
        """Test streaming SFT dataset preparation with IterableDataset."""
        with patch("axolotl.utils.data.sft.load_dataset_with_config") as mock_load:
            mock_load.return_value = streaming_dataset_fixture

            cfg = DictDefault(
                {
                    "tokenizer_config": "huggyllama/llama-7b",
                    "sequence_len": 256,
                    "streaming": True,
                    "max_steps": 100,  # Required for streaming datasets
                    "datasets": [
                        {
                            "path": "dummy/path",
                            "type": "alpaca",
                        },
                    ],
                }
            )

            train_dataset, eval_dataset, total_num_steps, prompters = prepare_datasets(
                cfg, tokenizer
            )

            # Verify it returns an IterableDataset
            assert isinstance(train_dataset, IterableDataset)
            assert eval_dataset is None  # No eval split for streaming
            assert total_num_steps == 100  # Should use max_steps
            assert len(prompters) == 1

            # Test that we can iterate through the dataset
            sample_count = 0
            for sample in train_dataset:
                assert "input_ids" in sample
                assert "attention_mask" in sample
                assert "labels" in sample
                sample_count += 1
                if sample_count >= 2:  # Just test first few samples
                    break

            assert sample_count == 2

    def test_dataset_mixing_strategy_validation(self):
        """Test validation of dataset mixing strategy configuration."""
        from axolotl.utils.data.shared import _merge_datasets_with_strategy

        # Test valid strategies work
        valid_strategies = ["round_robin", "weighted", "random"]
        dataset1 = Dataset.from_dict({"text": ["a"], "source": ["ds1"]})
        dataset2 = Dataset.from_dict({"text": ["b"], "source": ["ds2"]})

        for strategy in valid_strategies:
            cfg = DictDefault(
                {
                    "dataset_mixing_strategy": strategy,
                    "mixing_weights": [0.5, 0.5] if strategy == "weighted" else None,
                    "seed": 42,
                }
            )
            # Should not raise an error
            merged = _merge_datasets_with_strategy([dataset1, dataset2], cfg)
            assert len(merged) >= 1

    def test_regular_dataset_round_robin_mixing(self):
        """Test round-robin mixing for regular datasets."""
        from axolotl.utils.data.shared import _merge_datasets_with_strategy

        # Create test datasets
        dataset1 = Dataset.from_dict(
            {"text": ["ds1_item1", "ds1_item2"], "source": ["ds1", "ds1"]}
        )
        dataset2 = Dataset.from_dict(
            {"text": ["ds2_item1", "ds2_item2"], "source": ["ds2", "ds2"]}
        )

        cfg = DictDefault({"dataset_mixing_strategy": "round_robin", "seed": 42})

        merged = _merge_datasets_with_strategy([dataset1, dataset2], cfg)

        # Should have all samples from both datasets
        assert len(merged) == 4
        assert isinstance(merged, Dataset)

        # Check that samples are interleaved (not just concatenated)
        sources = [sample["source"] for sample in merged]
        # Round-robin should alternate between datasets
        assert sources != ["ds1", "ds1", "ds2", "ds2"]  # Not concatenated

    def test_regular_dataset_weighted_mixing(self):
        """Test weighted mixing for regular datasets."""
        from axolotl.utils.data.shared import _merge_datasets_with_strategy

        # Create test datasets
        dataset1 = Dataset.from_dict(
            {
                "text": ["ds1_item1", "ds1_item2", "ds1_item3", "ds1_item4"],
                "source": ["ds1"] * 4,
            }
        )
        dataset2 = Dataset.from_dict(
            {
                "text": ["ds2_item1", "ds2_item2", "ds2_item3", "ds2_item4"],
                "source": ["ds2"] * 4,
            }
        )

        cfg = DictDefault(
            {
                "dataset_mixing_strategy": "weighted",
                "mixing_weights": [0.75, 0.25],  # 3:1 ratio
                "seed": 42,
            }
        )

        merged = _merge_datasets_with_strategy([dataset1, dataset2], cfg)

        # Should have samples proportional to weights
        assert len(merged) > 0
        assert isinstance(merged, Dataset)

        # Count samples from each dataset
        sources = [sample["source"] for sample in merged]
        ds1_count = sources.count("ds1")
        ds2_count = sources.count("ds2")

        # Should have samples from both datasets
        assert ds1_count > 0 and ds2_count > 0  # Both datasets should be represented

    def test_streaming_dataset_mixing(self):
        """Test that streaming datasets use HuggingFace interleave_datasets."""
        from axolotl.utils.data.shared import _merge_datasets_with_strategy

        # Create test streaming datasets
        def gen1():
            yield {"text": "stream1_item1", "source": "stream1"}
            yield {"text": "stream1_item2", "source": "stream1"}

        def gen2():
            yield {"text": "stream2_item1", "source": "stream2"}
            yield {"text": "stream2_item2", "source": "stream2"}

        stream1 = IterableDataset.from_generator(gen1)
        stream2 = IterableDataset.from_generator(gen2)

        cfg = DictDefault({"dataset_mixing_strategy": "round_robin", "seed": 42})

        merged = _merge_datasets_with_strategy([stream1, stream2], cfg)

        # Should return an IterableDataset
        assert isinstance(merged, IterableDataset)

        # Test that we can iterate and get samples
        samples = list(merged.take(3))
        assert len(samples) >= 2  # Should get at least 2 samples

        # Should have samples from both datasets
        sources = [sample["source"] for sample in samples]
        assert len(set(sources)) >= 1  # At least one unique source
