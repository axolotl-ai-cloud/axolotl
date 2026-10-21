"""
Simple end-to-end smoke tests for FP8 mixed precision training
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists


@pytest.mark.skip(
    reason="test_cut_cross_entropy.py leaks its LlamaForCausalLM patch into later tests "
    "(cleanup_monkeypatches misses it on first import), so these fp32 runs hit CCE's "
    "bf16-only backward"
)
class TestFP8Integration:
    """
    e2e smoke tests for FP8 mixed precision training with Axolotl
    """

    def test_fp8_single_gpu_smoke(self, temp_dir):
        """Smoke test for single GPU FP8 + torch.compile training"""

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "trust_remote_code": True,
                "sequence_len": 512,
                "val_set_size": 0.05,
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
                "max_steps": 3,  # Very short smoke test
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "sdp_attention": True,
                "pad_to_sequence_len": True,
                "sample_packing": True,
                "fp8": True,
                "torch_compile": True,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
