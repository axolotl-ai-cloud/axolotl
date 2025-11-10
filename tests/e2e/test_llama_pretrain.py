"""E2E tests for llama pretrain"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard


class TestPretrainLlama:
    """Test case for Llama models w pretraining"""

    @pytest.mark.parametrize(
        ("sample_packing", "pretrain_multipack_attn"),
        [
            (False, False),
            (True, True),
            (True, False),
        ],
    )
    def test_pretrain(self, temp_dir, sample_packing, pretrain_multipack_attn):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "flash_attention": True,
                "sequence_len": 1024,
                "sample_packing": sample_packing,
                "pretrain_multipack_attn": pretrain_multipack_attn,
                "dataset_num_proc": 1,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "pretraining_dataset": [
                    {
                        "path": "allenai/c4",
                        "name": "en",
                        "type": "pretrain",
                    }
                ],
                "max_steps": 5,
                "num_epochs": 1,
                "micro_batch_size": 2,
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
        loss_threshold = 3.6
        if sample_packing and not pretrain_multipack_attn:
            loss_threshold = 6.5
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            loss_threshold,
            "Train Loss (%s) is too high",
        )
