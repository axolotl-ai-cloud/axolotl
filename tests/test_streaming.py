"""Test streaming configuration and data loading functionality."""

import unittest
from unittest.mock import Mock, patch

from datasets import IterableDataset

from axolotl.utils.config import validate_config
from axolotl.utils.data.sft import (
    _prepare_streaming_dataset,
    prepare_datasets,
)
from axolotl.utils.dict import DictDefault


class TestStreamingConfig(unittest.TestCase):
    """Test streaming configuration and deprecation handling."""

    def test_streaming_multipack_buffer_size_deprecation(self):
        """Test that pretrain_multipack_buffer_size is properly deprecated."""
        # Test with old config name
        cfg_old = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "pretrain_multipack_buffer_size": 5000,
                "datasets": [{"path": "test/dataset", "type": "alpaca"}],
                "sequence_len": 256,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
            }
        )

        with self.assertLogs("axolotl.utils.schemas.validation", level="WARNING") as cm:
            validated_cfg = validate_config(cfg_old)
            self.assertIn("pretrain_multipack_buffer_size` is deprecated", cm.output[0])

        self.assertEqual(validated_cfg.streaming_multipack_buffer_size, 5000)
        self.assertIsNone(
            getattr(validated_cfg, "pretrain_multipack_buffer_size", None)
        )

    def test_streaming_multipack_buffer_size_new(self):
        """Test that new streaming_multipack_buffer_size works correctly."""
        cfg_new = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "streaming_multipack_buffer_size": 7000,
                "datasets": [{"path": "test/dataset", "type": "alpaca"}],
                "sequence_len": 256,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
            }
        )

        validated_cfg = validate_config(cfg_new)
        self.assertEqual(validated_cfg.streaming_multipack_buffer_size, 7000)

    def test_both_buffer_sizes_raises_error(self):
        """Test that having both old and new buffer size configs raises an error."""
        cfg_both = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "pretrain_multipack_buffer_size": 5000,
                "streaming_multipack_buffer_size": 7000,
                "datasets": [{"path": "test/dataset", "type": "alpaca"}],
                "sequence_len": 256,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
            }
        )

        with self.assertRaises(ValueError) as cm:
            validate_config(cfg_both)
        self.assertIn("both are set", str(cm.exception))


class TestStreamingDatasetPreparation(unittest.TestCase):
    """Test dataset preparation with streaming configuration."""

    def setUp(self):
        self.tokenizer = Mock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1

    @patch("axolotl.utils.data.sft._prepare_streaming_dataset")
    def test_prepare_datasets_with_streaming_true(self, mock_prepare_streaming):
        """Test that streaming=True triggers streaming dataset preparation."""
        cfg = DictDefault(
            {
                "streaming": True,
                "datasets": [{"path": "test/dataset", "type": "alpaca"}],
            }
        )

        mock_prepare_streaming.return_value = (Mock(), None, 100, [])

        prepare_datasets(cfg, self.tokenizer)

        mock_prepare_streaming.assert_called_once_with(cfg, self.tokenizer, None)

    @patch("axolotl.utils.data.sft._prepare_streaming_dataset")
    def test_prepare_datasets_with_pretraining_dataset(self, mock_prepare_streaming):
        """Test that pretraining_dataset triggers streaming dataset preparation."""
        cfg = DictDefault(
            {
                "pretraining_dataset": "test/dataset",
            }
        )

        mock_prepare_streaming.return_value = (Mock(), None, 100, [])

        prepare_datasets(cfg, self.tokenizer)

        mock_prepare_streaming.assert_called_once_with(cfg, self.tokenizer, None)

    @patch("axolotl.utils.data.sft._prepare_standard_dataset")
    def test_prepare_datasets_without_streaming(self, mock_prepare_standard):
        """Test that without streaming, standard dataset preparation is used."""
        cfg = DictDefault(
            {
                "datasets": [{"path": "test/dataset", "type": "alpaca"}],
            }
        )

        mock_prepare_standard.return_value = (Mock(), None, 100, [])

        prepare_datasets(cfg, self.tokenizer)

        mock_prepare_standard.assert_called_once_with(cfg, self.tokenizer, None)


class TestStreamingWithSamplePacking(unittest.TestCase):
    """Test streaming dataset preparation with sample packing."""

    def setUp(self):
        self.tokenizer = Mock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1

    @patch("axolotl.utils.data.sft._load_streaming_dataset")
    def test_streaming_sft_with_sample_packing_sets_split(self, mock_load_streaming):
        """Test that streaming SFT with sample_packing sets default split."""
        cfg = DictDefault(
            {
                "streaming": True,
                "sample_packing": True,
                "datasets": [{"path": "test/dataset", "type": "alpaca"}],
                "sequence_len": 256,
                "micro_batch_size": 1,
            }
        )

        mock_load_streaming.return_value = Mock(spec=IterableDataset)

        with patch("axolotl.utils.data.sft._load_and_prepare_datasets"):
            _prepare_streaming_dataset(cfg, self.tokenizer, None)

            # Check that the dataset config has split set to 'train'
            call_args = mock_load_streaming.call_args
            dataset_config = call_args[0][0]
            self.assertEqual(dataset_config.split, "train")

    def test_multipack_attn_forced_true_for_sft(self):
        """Test that multipack_attn is forced to True for SFT with sample packing."""
        from axolotl.utils.data.streaming import wrap_streaming_dataset

        cfg = DictDefault(
            {
                "sample_packing": True,
                "pretrain_multipack_attn": False,  # Should be overridden for SFT
                "pretraining_dataset": None,  # This makes it SFT
                "sequence_len": 256,
                "micro_batch_size": 1,
                "streaming_multipack_buffer_size": 1000,
                "seed": 42,
            }
        )

        mock_dataset = Mock()
        mock_dataset.features = None  # For streaming datasets
        mock_dataset.__iter__ = Mock(return_value=iter([]))  # Empty iterator
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_ds_wrapper = Mock()

        with patch(
            "axolotl.utils.data.streaming.PretrainingBatchSamplerDataCollatorForSeq2Seq"
        ) as mock_collator:
            with patch("axolotl.utils.data.streaming.encode_packed_streaming"):
                wrap_streaming_dataset(
                    mock_dataset, self.tokenizer, cfg, mock_ds_wrapper
                )

                # Check that multipack_attn=True was used in the collator
                mock_collator.assert_called_once()
                call_kwargs = mock_collator.call_args[1]
                self.assertTrue(call_kwargs["multipack_attn"])

    def test_multipack_attn_respects_config_for_pretraining(self):
        """Test that multipack_attn respects config for pretraining datasets."""
        from axolotl.utils.data.streaming import wrap_streaming_dataset

        cfg = DictDefault(
            {
                "sample_packing": True,
                "pretrain_multipack_attn": False,  # Should be respected for pretraining
                "pretraining_dataset": "test/dataset",  # This makes it pretraining
                "sequence_len": 256,
                "micro_batch_size": 1,
                "streaming_multipack_buffer_size": 1000,
                "seed": 42,
            }
        )

        mock_dataset = Mock()
        mock_dataset.features = None  # For streaming datasets
        mock_dataset.__iter__ = Mock(return_value=iter([]))  # Empty iterator
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_ds_wrapper = Mock()

        with patch(
            "axolotl.utils.data.streaming.PretrainingBatchSamplerDataCollatorForSeq2Seq"
        ) as mock_collator:
            with patch("axolotl.utils.data.streaming.encode_packed_streaming"):
                wrap_streaming_dataset(
                    mock_dataset, self.tokenizer, cfg, mock_ds_wrapper
                )

                # Check that multipack_attn=False was used (respecting config)
                mock_collator.assert_called_once()
                call_kwargs = mock_collator.call_args[1]
                self.assertFalse(call_kwargs["multipack_attn"])


if __name__ == "__main__":
    unittest.main()
