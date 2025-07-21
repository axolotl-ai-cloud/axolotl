"""
Simple end-to-end smoke tests for FP8 mixed precision training
"""

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists, require_torch_2_4_1


class FP8IntegrationTestCase:
    """
    e2e smoke tests for FP8 mixed precision training with Axolotl
    """

    @require_torch_2_4_1
    def test_fp8_single_gpu_smoke(self, temp_dir):
        """Smoke test for single GPU FP8 + torch.compile training"""
        # pylint: disable=duplicate-code
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
                "optimizer": "adamw_torch_fused",  # Use standard optimizer for stability
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "fp8": True,  # Enable FP8 mixed precision
                "torch_compile": True,  # Essential for FP8 performance
                "save_safetensors": True,
                "save_first_step": False,
            }
        )

        # pylint: disable=duplicate-code
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
