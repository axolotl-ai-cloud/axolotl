"""
unit tests for axolotl.core.trainer_builder
"""

import pytest

from axolotl.core.trainer_builder import HFRLTrainerBuilder
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer


@pytest.fixture(name="cfg")
def fixture_cfg():
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.00005,
            "save_steps": 100,
            "output_dir": "./model-out",
            "warmup_steps": 10,
            "gradient_checkpointing": False,
            "optimizer": "adamw_torch_fused",
            "sequence_len": 2048,
            "rl": True,
            "adam_beta1": 0.998,
            "adam_beta2": 0.9,
            "adam_epsilon": 0.00001,
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": True,
            "model_config_type": "llama",
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
        }
    )

    normalize_config(cfg)

    return cfg


@pytest.fixture(name="tokenizer")
def fixture_tokenizer(cfg):
    return load_tokenizer(cfg)


@pytest.fixture(name="model")
def fixture_model(cfg, tokenizer):
    return load_model(cfg, tokenizer)


class TestHFRLTrainerBuilder:
    """
    TestCase class for DPO trainer builder
    """

    def test_build_training_arguments(self, cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(cfg, model, tokenizer)
        training_arguments = builder.build_training_arguments(100)
        assert training_arguments.adam_beta1 == 0.998
        assert training_arguments.adam_beta2 == 0.9
        assert training_arguments.adam_epsilon == 0.00001
        assert training_arguments.dataloader_num_workers == 1
        assert training_arguments.dataloader_pin_memory is True
