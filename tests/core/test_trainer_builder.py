"""
unit tests for axolotl.core.trainer_builder
"""
import pytest

from axolotl.core.trainer_builder import HFDPOTrainerBuilder
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer


@pytest.fixture(name="cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "LlamaTokenizer",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.00005,
            "save_steps": 100,
            "output_dir": "./model-out",
            "warmup_steps": 10,
            "gradient_checkpointing": False,
            "optimizer": "adamw_torch",
            "sequence_len": 2048,
            "rl": True,
            "adam_beta1": 0.998,
            "adam_beta2": 0.9,
            "adam_epsilon": 0.00001,
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": True,
        }
    )


@pytest.fixture(name="tokenizer")
def fixture_tokenizer(cfg):
    return load_tokenizer(cfg)


@pytest.fixture(name="model")
def fixture_model(cfg, tokenizer):
    return load_model(cfg, tokenizer)


class TestHFDPOTrainerBuilder:
    """
    TestCase class for DPO trainer builder
    """

    def test_build_training_arguments(self, cfg, model, tokenizer):
        builder = HFDPOTrainerBuilder(cfg, model, tokenizer)
        training_arguments = builder.build_training_arguments(100)
        assert training_arguments.adam_beta1 == 0.998
        assert training_arguments.adam_beta2 == 0.9
        assert training_arguments.adam_epsilon == 0.00001
        assert training_arguments.dataloader_num_workers == 1
        assert training_arguments.dataloader_pin_memory is True
