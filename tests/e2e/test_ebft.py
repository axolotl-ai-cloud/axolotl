"""E2E tests for EBFT"""

import unittest
from pathlib import Path

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_preference_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, with_temp_dir


class TestEBFTLlamaLora(unittest.TestCase):
    """
    Test case for EBFT strided mode, which generates in-process and needs no vLLM server
    """

    @with_temp_dir
    def test_ebft_strided_lora(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 512,
                "adapter": "lora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "rl": "ebft",
                "ebft": {
                    "mode": "strided",
                    "stride": 8,
                    "context_length": 8,
                    "generate_max_len": 8,
                    "n_samples_per_prompt": 2,
                    "feature_layers": [0.5],
                    "embed_method": "last_token",
                    "advantage_estimator": "rloo",
                    "min_completion_prefix": 8,
                },
                "datasets": [
                    {
                        "path": "nvidia/OpenCodeInstruct",
                        "type": "ebft_strided_structured.transform",
                        "split": "train[:1%]",
                    },
                ],
                "val_set_size": 0.0,
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.000001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_steps": 5,
                "warmup_steps": 1,
                "bf16": "auto",
                "attn_implementation": "flex_attention",
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": True},
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-5", cfg)
