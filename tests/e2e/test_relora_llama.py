"""
E2E tests for relora llama
"""

import logging
import os
import unittest
from pathlib import Path

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import check_tensorboard, with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestReLoraLlama(unittest.TestCase):
    """
    Test case for Llama models using LoRA
    """

    @with_temp_dir
    def test_relora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 2048,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "flash_attention": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj"],
                "relora_steps": 100,
                "relora_warmup_steps": 20,
                "relora_anneal_steps": 10,
                "relora_prune_ratio": 0.9,
                "relora_cpu_offload": True,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "warmup_steps": 20,
                "num_epochs": 2,
                "max_steps": 205,  # at least 2x relora_steps
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "use_tensorboard": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (
            Path(temp_dir) / "checkpoint-100/adapter/adapter_model.safetensors"
        ).exists()
        assert (Path(temp_dir) / "checkpoint-100/relora/model.safetensors").exists()

        check_tensorboard(
            temp_dir + "/runs", "train/grad_norm", 0.2, "grad_norm is too high"
        )
