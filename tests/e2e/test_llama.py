"""
E2E tests for llama
"""

import logging
import os

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestLlama:
    """
    Test case for Llama models
    """

    def test_fft_trust_remote_code(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "tokenizer_type": "LlamaTokenizer",
                "trust_remote_code": True,
                "sequence_len": 512,
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
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": True,
                "save_safetensors": True,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    def test_fix_untrained_tokens(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "fix_untrained_tokens": True,
                "sequence_len": 512,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                    "bos_token": "<|custom_im_start|>",
                    "eos_token": "<|custom_im_end|>",
                },
                "datasets": [
                    {
                        "chat_template": "jinja",
                        "chat_template_jinja": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|custom_im_start|>' + message['role'] + '\n' + message['content'] + '<|custom_im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|custom_im_start|>assistant\n' }}{% endif %}",
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": True,
                "save_safetensors": True,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    def test_fix_untrained_tokens_already_trained(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "fix_untrained_tokens": True,
                "sequence_len": 512,
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
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": True,
                "save_safetensors": True,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    def test_batch_flattening(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "trust_remote_code": True,
                "sequence_len": 512,
                "val_set_size": 0.01,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": False,
                "batch_flattening": True,
                "bf16": True,
                "save_safetensors": True,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
