"""E2E smoke test for diffusion training plugin."""

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists


class TestDiffusion:
    """Test case for diffusion training plugin."""

    def test_diffusion_smoke_test(self, temp_dir):
        """
        Smoke test for diffusion training to ensure the plugin loads and trains without
        error.
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
                "max_steps": 3,
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
                # Diffusion-specific config
                "plugins": ["axolotl.integrations.diffusion.DiffusionPlugin"],
                "diffusion": {
                    # sample generation
                    "generate_samples": True,
                    "generation_interval": 1,
                    "num_generation_samples": 1,
                    "generation_steps": 2,
                    "generation_max_length": 32,
                    "generation_temperature": 0.0,
                    # training-specific
                    "mask_token_id": 16,
                    "eps": 1e-3,
                    "importance_weighting": False,
                },
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    def test_diffusion_sft_labels(self, temp_dir):
        """Test that diffusion training properly handles SFT data with labels."""
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
                "max_steps": 3,
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
                # Diffusion-specific config
                "plugins": ["axolotl.integrations.diffusion.DiffusionPlugin"],
                "diffusion": {
                    # sample generation
                    "generate_samples": True,
                    "generation_interval": 1,
                    "num_generation_samples": 1,
                    "generation_steps": 2,
                    "generation_max_length": 32,
                    "generation_temperature": 0.0,
                    # training-specific
                    "mask_token_id": 16,
                    "eps": 1e-3,
                    "importance_weighting": True,
                },
                # Ensure we have proper SFT labels
                "train_on_inputs": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        # Verify that the dataset has labels
        sample = dataset_meta.train_dataset[0]
        assert "labels" in sample, "SFT dataset should have labels"

        # Check that some labels are -100 (prompt tokens)
        labels = sample["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        assert -100 in labels, "SFT dataset should have -100 labels for prompt tokens"

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
