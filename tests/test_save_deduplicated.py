"""Tests to verify that deduplication runs before dataset saving during preprocessing.

This addresses GitHub issue #2719: Save De-duplicated Set During Pre-processing.
"""

from unittest.mock import MagicMock, call, patch

import pytest
from datasets import Dataset

from axolotl.utils.dict import DictDefault


class TestSFTSaveDeduplicatedBeforeSave:
    """Verify that in SFT data loading, deduplication occurs before saving."""

    @patch("axolotl.utils.data.sft.save_preprocessed_dataset")
    @patch("axolotl.utils.data.sft.generate_dataset_hash_from_config")
    @patch("axolotl.utils.data.sft.deduplicate_and_log_datasets")
    @patch("axolotl.utils.data.sft.merge_datasets")
    @patch("axolotl.utils.data.sft._load_and_process_single_dataset")
    @patch("axolotl.utils.data.sft.datasets_with_name_generator")
    def test_dedup_called_before_save_sft(
        self,
        mock_datasets_gen,
        mock_load_single,
        mock_merge,
        mock_dedup,
        mock_gen_hash,
        mock_save,
    ):
        """Deduplication should be called before save_preprocessed_dataset in SFT."""
        from axolotl.utils.data.sft import _load_raw_datasets

        # Set up mock data
        dataset = Dataset.from_dict({"text": ["a", "b", "a"], "label": [1, 2, 1]})
        deduped_dataset = Dataset.from_dict({"text": ["a", "b"], "label": [1, 2]})

        mock_datasets_gen.return_value = [DictDefault({"path": "test", "type": "alpaca"})]
        mock_load_single.return_value = (dataset, None)
        mock_merge.return_value = dataset
        mock_dedup.return_value = (deduped_dataset, None)
        mock_gen_hash.return_value = "testhash"

        cfg = DictDefault({
            "skip_prepare_dataset": False,
            "dataset_exact_deduplication": True,
            "sequence_len": 1024,
            "eval_sequence_len": None,
            "sample_packing": False,
            "is_preprocess": False,
            "seed": 42,
            "datasets": [{"path": "test", "type": "alpaca"}],
        })

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"

        # Track call order
        call_order = []
        mock_dedup.side_effect = lambda **kwargs: (
            call_order.append("dedup") or (deduped_dataset, None)
        )
        mock_save.side_effect = lambda *args, **kwargs: call_order.append("save")

        _load_raw_datasets(
            cfg=cfg,
            datasets_configs=cfg.datasets,
            tokenizer=tokenizer,
            split="train",
        )

        # Verify dedup was called
        assert "dedup" in call_order, "Deduplication should have been called"
        # Verify save was called
        assert "save" in call_order, "Save should have been called"
        # Verify dedup happened before save
        assert call_order.index("dedup") < call_order.index("save"), (
            "Deduplication must occur before saving the dataset"
        )

    @patch("axolotl.utils.data.sft.save_preprocessed_dataset")
    @patch("axolotl.utils.data.sft.generate_dataset_hash_from_config")
    @patch("axolotl.utils.data.sft.merge_datasets")
    @patch("axolotl.utils.data.sft._load_and_process_single_dataset")
    @patch("axolotl.utils.data.sft.datasets_with_name_generator")
    def test_no_dedup_when_disabled_sft(
        self,
        mock_datasets_gen,
        mock_load_single,
        mock_merge,
        mock_gen_hash,
        mock_save,
    ):
        """Deduplication should not be called when dataset_exact_deduplication is False."""
        from axolotl.utils.data.sft import _load_raw_datasets

        dataset = Dataset.from_dict({"text": ["a", "b", "a"], "label": [1, 2, 1]})

        mock_datasets_gen.return_value = [DictDefault({"path": "test", "type": "alpaca"})]
        mock_load_single.return_value = (dataset, None)
        mock_merge.return_value = dataset
        mock_gen_hash.return_value = "testhash"

        cfg = DictDefault({
            "skip_prepare_dataset": False,
            "dataset_exact_deduplication": False,
            "sequence_len": 1024,
            "eval_sequence_len": None,
            "sample_packing": False,
            "is_preprocess": False,
            "seed": 42,
            "datasets": [{"path": "test", "type": "alpaca"}],
        })

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"

        with patch("axolotl.utils.data.sft.deduplicate_and_log_datasets") as mock_dedup:
            _load_raw_datasets(
                cfg=cfg,
                datasets_configs=cfg.datasets,
                tokenizer=tokenizer,
                split="train",
            )
            mock_dedup.assert_not_called()


class TestRLSaveDeduplicatedBeforeSave:
    """Verify that in RL data loading, deduplication occurs before saving."""

    @patch("axolotl.utils.data.rl.save_preprocessed_dataset")
    @patch("axolotl.utils.data.rl.generate_dataset_hash_from_config")
    @patch("axolotl.utils.data.rl.deduplicate_and_log_datasets")
    @patch("axolotl.utils.data.rl.merge_datasets")
    @patch("axolotl.utils.data.rl.load_dataset_with_config")
    @patch("axolotl.utils.data.rl.datasets_with_name_generator")
    @patch("axolotl.loaders.load_tokenizer")
    def test_dedup_called_before_save_rl(
        self,
        mock_load_tokenizer,
        mock_datasets_gen,
        mock_load_dataset,
        mock_merge,
        mock_dedup,
        mock_gen_hash,
        mock_save,
    ):
        """Deduplication should be called before save_preprocessed_dataset in RL."""
        from axolotl.utils.data.rl import _load_split

        dataset = Dataset.from_dict({
            "prompt": ["hi", "bye", "hi"],
            "chosen": ["a", "b", "a"],
            "rejected": ["c", "d", "c"],
        })
        deduped_dataset = Dataset.from_dict({
            "prompt": ["hi", "bye"],
            "chosen": ["a", "b"],
            "rejected": ["c", "d"],
        })

        mock_datasets_gen.return_value = [DictDefault({"path": "test", "type": None})]
        mock_load_dataset.return_value = dataset
        mock_merge.return_value = dataset
        mock_dedup.return_value = (deduped_dataset, None)
        mock_gen_hash.return_value = "testhash"

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"
        mock_load_tokenizer.return_value = tokenizer

        cfg = DictDefault({
            "skip_prepare_dataset": False,
            "dataset_exact_deduplication": True,
            "sequence_len": 1024,
            "rl": "dpo",
            "datasets": [{"path": "test", "type": None}],
            "hf_use_auth_token": False,
            "dataset_num_proc": 1,
            "is_preprocess": False,
        })

        call_order = []
        mock_dedup.side_effect = lambda **kwargs: (
            call_order.append("dedup") or (deduped_dataset, None)
        )
        mock_save.side_effect = lambda *args, **kwargs: call_order.append("save")

        _load_split(cfg, split="train")

        assert "dedup" in call_order, "Deduplication should have been called"
        assert "save" in call_order, "Save should have been called"
        assert call_order.index("dedup") < call_order.index("save"), (
            "Deduplication must occur before saving the dataset"
        )
