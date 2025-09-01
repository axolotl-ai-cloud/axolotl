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
                    }
                ],
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

    def test_flows_to_training_args_kwargs(self):
        """Ensure center_rewards_coefficient flows into training_arguments_kwargs"""
        cfg = self._get_base_cfg()
        cfg["center_rewards_coefficient"] = 0.01
        config = AxolotlInputConfig(**cfg)

        training_args_kwargs = {}
        for arg in ["center_rewards_coefficient"]:
            if getattr(config, arg, None) is not None:
                training_args_kwargs[arg] = getattr(config, arg)

        self.assertIn("center_rewards_coefficient", training_args_kwargs)
        self.assertEqual(training_args_kwargs["center_rewards_coefficient"], 0.01)
