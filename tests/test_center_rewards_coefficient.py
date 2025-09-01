import unittest
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.config import AxolotlInputConfig


class CenterRewardsIntegrationTest(unittest.TestCase):
    def _get_base_cfg(self):
        return DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "model_type": "AutoModelForSequenceClassification",
                "num_labels": 1,
                "tokenizer_type": "AutoTokenizer",
                "reward_model": True,
                "datasets": [
                    {
                        "path": "argilla/distilabel-intel-orca-dpo-pairs",
                        "type": "bradley_terry.chat_template",
                        "split": "train[:1%]",
                    }
                ],
                "chat_template": "chatml",
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
                "output_dir": "./test_output",
            }
        )

    def test_schema_accepts_center_rewards_coefficient(self):
        """Ensure schema accepts and validates center_rewards_coefficient"""
        cfg = self._get_base_cfg()
        cfg["center_rewards_coefficient"] = 0.01
        config = AxolotlInputConfig(**cfg)
        self.assertEqual(config.center_rewards_coefficient, 0.01)

    def test_schema_defaults_to_none(self):
        """Ensure center_rewards_coefficient defaults to None when not set"""
        config = AxolotlInputConfig(**self._get_base_cfg())
        self.assertIsNone(config.center_rewards_coefficient)

    def test_flows_to_training_arguments(self):
        """Test that center_rewards_coefficient flows through trainer builder to training arguments"""
        from axolotl.common.datasets import load_datasets
        from axolotl.core.builders import HFCausalTrainerBuilder
        from axolotl.loaders import ModelLoader, load_tokenizer
        from axolotl.utils.config import normalize_config
        
        cfg = self._get_base_cfg()
        cfg["center_rewards_coefficient"] = 0.01
        # Add required fields to prevent errors
        cfg.update({
            "sequence_len": 2048,
            "model_config_type": "llama",
            "val_set_size": 0,
            "num_epochs": 1,
            "max_steps": 100,
            "save_steps": 100,
            "eval_steps": 50,
            "logging_steps": 10,
            "seed": 42,
            "dataset_processes": 4,
            "context_parallel_size": 1,
            "tensor_parallel_size": 1,
            "optimizer": "adamw_torch",
        })
        normalize_config(cfg)
        
        # Load tokenizer and model
        tokenizer = load_tokenizer(cfg)
        model, _ = ModelLoader(cfg, tokenizer).load()
        
        # Load datasets
        dataset_meta = load_datasets(cfg=cfg)
        
        # Create builder and set datasets
        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        builder.train_dataset = dataset_meta.train_dataset
        builder.eval_dataset = dataset_meta.eval_dataset
        
        # Build trainer and check training arguments
        trainer = builder.build(100)
        
        # Verify center_rewards_coefficient is in training arguments
        self.assertTrue(hasattr(trainer.args, "center_rewards_coefficient"))
        self.assertEqual(trainer.args.center_rewards_coefficient, 0.01)
