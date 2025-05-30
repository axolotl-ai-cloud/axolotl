"""
E2E tests for QAT
"""

import unittest
from pathlib import Path

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestQATLlama(unittest.TestCase):
    """
    Test case for QAT Llama models
    """

    @with_temp_dir
    def test_qat_lora(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "message_property_mappings": {
                            "role": "from",
                            "content": "value",
                        },
                        "drop_system_message": True,
                        "split": "train[:1%]",
                    },
                ],
                "chat_template": "chatml",
                "qat": {
                    "quantize_embedding": True,
                    "activation_dtype": "int8",
                    "weight_dtype": "int8",
                    "group_size": 8,
                },
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_safetensors": True,
                "bf16": True,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-5", cfg)
