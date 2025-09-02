"""E2E tests for streaming dataset functionality"""

# pylint: disable=duplicate-code

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard


class TestStreamingDatasets:
    """Test case for streaming datasets"""

    @pytest.mark.parametrize(
        "sample_packing",
        [True, False],
    )
    def test_streaming_dataset(self, temp_dir, sample_packing):
        """Test streaming datasets"""

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 1024,
                "sample_packing": sample_packing,
                "pretrain_multipack_attn": sample_packing,
                "streaming_multipack_buffer_size": 10000,
                "dataset_processes": 1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                # Streaming config
                "streaming": True,
                "max_steps": 3,
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

        # Verify training actually happened by checking loss decrease
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            3.0,
            "Train Loss (%s) is too high",
        )
