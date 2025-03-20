"""
tests for loading loras
"""

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer

# pylint: disable=duplicate-code
minimal_config = DictDefault(
    {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "learning_rate": 0.000001,
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            }
        ],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
)


class TestLoRALoad:
    """
    Test class for loading LoRA weights
    """

    def test_load_lora_weights(self):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "sequence_len": 1024,
            }
            | minimal_config
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        tokenizer = load_tokenizer(cfg)
        load_model(cfg, tokenizer)

    def test_load_lora_weights_empty_dropout(self):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": None,
                "lora_target_linear": True,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "sequence_len": 1024,
            }
            | minimal_config
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        assert cfg.lora_dropout == 0.0
        tokenizer = load_tokenizer(cfg)
        load_model(cfg, tokenizer)
