"""
E2E smoke test for diffusion training plugin
"""

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists


class TestDiffusion:
    """
    Test case for diffusion training plugin
    """

    def test_diffusion_smoke_test(self, temp_dir):
        """
        Smoke test for diffusion training to ensure the plugin loads and trains without error.
        """
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "trust_remote_code": True,
                "sequence_len": 256,
                "val_set_size": 0.1,
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
                "max_steps": 3,  # Very short for smoke test
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "bf16": True,
                "save_safetensors": True,
                "save_first_step": False,
                "logging_steps": 1,
                "eval_steps": 3,
                "plugins": ["axolotl.integrations.diffusion.DiffusionPlugin"],
                # Diffusion-specific config
                "diffusion_mask_token_id": 32000,
                "diffusion_eps": 1e-3,
                "diffusion_importance_weighting": False,
            }
        )

        # Normalize and validate config
        cfg = normalize_config(cfg)
        cfg = validate_config(cfg)

        # Load datasets to ensure they work with diffusion training
        datasets_meta = load_datasets(cfg=cfg, cli_args=DictDefault({}))
        assert datasets_meta.train_dataset is not None
        assert len(datasets_meta.train_dataset) > 0

        # Run training
        train(cfg=cfg, cli_args=DictDefault({}), dataset_meta=datasets_meta)

        # Check that model was saved
        check_model_output_exists(cfg)

    def test_diffusion_sft_labels(self, temp_dir):
        """
        Test that diffusion training properly handles SFT data with labels.
        """
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "trust_remote_code": True,
                "sequence_len": 256,
                "val_set_size": 0.1,
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
                "max_steps": 2,  # Very short for smoke test
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "bf16": True,
                "save_safetensors": True,
                "save_first_step": False,
                "logging_steps": 1,
                "eval_steps": 2,
                "plugins": ["axolotl.integrations.diffusion.DiffusionPlugin"],
                # Diffusion-specific config
                "diffusion_mask_token_id": 32000,
                "diffusion_eps": 1e-3,
                "diffusion_importance_weighting": True,  # Test importance weighting
                # Ensure we have proper SFT labels
                "train_on_inputs": False,  # This ensures prompt tokens get -100 labels
            }
        )

        # Normalize and validate config
        cfg = normalize_config(cfg)
        cfg = validate_config(cfg)

        # Load datasets
        datasets_meta = load_datasets(cfg=cfg, cli_args=DictDefault({}))
        
        # Verify that the dataset has labels
        sample = datasets_meta.train_dataset[0]
        assert "labels" in sample, "SFT dataset should have labels"
        
        # Check that some labels are -100 (prompt tokens)
        labels = sample["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        assert -100 in labels, "SFT dataset should have -100 labels for prompt tokens"

        # Run training
        train(cfg=cfg, cli_args=DictDefault({}), dataset_meta=datasets_meta)

        # Check that model was saved
        check_model_output_exists(cfg)