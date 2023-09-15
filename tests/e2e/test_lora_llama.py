"""
E2E tests for lora llama
"""

import logging
import os
import tempfile
import unittest

from axolotl.common.cli import TrainerCliArgs
from axolotl.train import TrainDatasetMeta, train
from axolotl.utils.config import normalize_config
from axolotl.utils.data import prepare_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_tokenizer

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


def load_datasets(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,  # pylint:disable=unused-argument
) -> TrainDatasetMeta:
    tokenizer = load_tokenizer(cfg)

    train_dataset, eval_dataset, total_num_steps = prepare_dataset(cfg, tokenizer)

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )


class TestLoraLlama(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    def test_lora(self):
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "base_model_config": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": tempfile.mkdtemp(),
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)

    def test_lora_packing(self):
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "base_model_config": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 2,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": tempfile.mkdtemp(),
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
