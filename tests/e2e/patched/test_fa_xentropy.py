"""
E2E tests for lora llama
"""

import logging
import os

import pytest
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists, check_tensorboard

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestFAXentropyLlama:
    """
    Test case for Llama models using LoRA w multipack
    """

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 4],
    )
    def test_lora_packing_fa_cross_entropy(self, temp_dir, gradient_accumulation_steps):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "flash_attn_cross_entropy": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "field_messages": "conversations",
                        "message_field_content": "value",
                        "message_field_role": "from",
                        "type": "chat_template",
                        "split": "train[:2%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "save_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "use_tensorboard": True,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        cfg = validate_config(cfg)
        normalize_config(cfg)

        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        check_tensorboard(
            temp_dir + "/runs", "train/train_loss", 1.5, "Train Loss is too high"
        )
