"""E2E tests for streaming dataset functionality"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard


class TestStreamingDatasets:
    """Test case for streaming datasets with different mixing strategies"""

    @pytest.mark.parametrize(
        ("dataset_mixing_strategy", "mixing_weights"),
        [
            ("round_robin", None),
            ("weighted", [0.7, 0.3]),
            ("random", None),
        ],
    )
    def test_streaming_dataset_mixing_strategies(
        self, temp_dir, dataset_mixing_strategy, mixing_weights
    ):
        """Test different mixing strategies with streaming datasets"""

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 1024,
                "sample_packing": False,
                "dataset_processes": 1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                # Streaming config
                "streaming": True,
                "max_steps": 3,  # Very small for smoke test
                "dataset_mixing_strategy": dataset_mixing_strategy,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
                "save_first_step": False,
            }
        )

        # Add mixing weights if specified
        if mixing_weights:
            cfg["mixing_weights"] = mixing_weights

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        # Verify training actually happened by checking loss decrease
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            25.0,  # Loss should be reasonable for a smoke test (higher threshold for streaming)
            "Train Loss (%s) is too high",
        )

    def test_streaming_eval_specific_mixing(self, temp_dir):
        """Test eval-specific mixing strategy override"""

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 512,
                "sample_packing": False,
                "dataset_processes": 1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "test_datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                        "split": "train",  # Specify train split for eval dataset
                    },
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train",  # Specify train split for eval dataset
                    },
                ],
                # Streaming config
                "streaming": True,
                "eval_streaming": True,
                "max_steps": 3,
                # Different mixing for train vs eval
                "dataset_mixing_strategy": "round_robin",
                "eval_dataset_mixing_strategy": "weighted",
                "eval_mixing_weights": [0.6, 0.4],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
                "save_first_step": False,
                "eval_steps": 3,  # Eval at the end
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        # Check both train and eval losses
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            25.0,
            "Train Loss (%s) is too high",
        )
        check_tensorboard(
            temp_dir + "/runs",
            "eval/eval_loss",
            25.0,
            "Eval Loss (%s) is too high",
        )

    def test_streaming_validation_error(self, temp_dir):
        """Test that pydantic validation catches invalid streaming configs"""

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                ],
                "streaming": True,
                "max_steps": 3,
                # Invalid: wrong number of weights for datasets
                "dataset_mixing_strategy": "weighted",
                "mixing_weights": [1.0],  # Should be [0.x, 0.y] for 2 datasets
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
            }
        )

        # This should raise a validation error
        with pytest.raises(Exception) as exc_info:
            validate_config(cfg)

        # Verify it's the right validation error
        assert "mixing_weights length" in str(exc_info.value)
        assert "must match number of datasets" in str(exc_info.value)

    def test_streaming_three_datasets_weighted(self, temp_dir):
        """Test weighted mixing with three datasets"""

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 512,
                "sample_packing": False,
                "dataset_processes": 1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                    },
                    {
                        "path": "yahma/alpaca-cleaned",
                        "type": "alpaca",
                    },
                ],
                # Streaming config
                "streaming": True,
                "max_steps": 3,
                "dataset_mixing_strategy": "weighted",
                "mixing_weights": [0.5, 0.3, 0.2],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "val_set_size": 0.0,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "use_tensorboard": True,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            25.0,
            "Train Loss (%s) is too high",
        )
