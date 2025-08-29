"""
e2e test for saving the tokenizer
"""

from unittest.mock import patch

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists


def test_tokenizer_no_save_jinja_files(temp_dir):
    # pylint: disable=duplicate-code
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "tokenizer_type": "AutoTokenizer",
            "sequence_len": 1024,
            "load_in_8bit": True,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_linear": True,
            "val_set_size": 0.02,
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
            "chat_template": "chatml",
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                },
            ],
            "num_epochs": 1,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "output_dir": temp_dir,
            "learning_rate": 0.00001,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "cosine",
            "max_steps": 5,
            "save_first_step": False,
            "fp16": False,
            "tokenizer_save_jinja_files": False,
        }
    )

    cfg = validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)

    with patch("axolotl.train.execute_training"):
        train(cfg=cfg, dataset_meta=dataset_meta)

    check_model_output_exists(temp_dir, cfg)
    with open(f"{temp_dir}/tokenizer_config.json", "r", encoding="utf-8") as f:
        tokenizer_config = f.read()
        assert "chat_template" in tokenizer_config
