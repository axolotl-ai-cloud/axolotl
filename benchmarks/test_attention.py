"""Benchmarks for various attention mechanisms"""

import gc

import pytest
import torch
from pytest_cases import parametrize_with_cases
from tabulate import tabulate  # type: ignore

from axolotl.utils.bench import gpu_memory_usage
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.validation import validate_config


class TestConfigs:  # pylint: disable=missing-class-docstring
    def cfg_llama2(self):
        return DictDefault(
            {
                "base_model": "meta-llama/Llama-2-7b-chat-hf",
                "base_model_config": "meta-llama/Llama-2-7b-chat-hf",
                "model_type": "LlamaForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "device_map": {"": 0},
                "adapter": None,
                "load_in_8bit": True,
            }
        )

    def cfg_llama2_xformers(self):
        return self.cfg_llama2() | DictDefault(
            {
                "xformers_attention": True,
            }
        )

    def cfg_llama2_flashattn(self):
        return self.cfg_llama2() | DictDefault(
            {
                "flash_attention": True,
            }
        )


@pytest.fixture(autouse=True)
def memory_cleanup():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@parametrize_with_cases("cfg", cases=TestConfigs, prefix="cfg_")
def test_benchmark_attn(cfg, results_bag):
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    results_bag.vram_baseline = gpu_memory_usage()
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)
    model = load_model(cfg, tokenizer)
    del tokenizer
    del model


def test_synthesis(module_results_df):
    module_results_df.drop(["cfg", "pytest_obj"], axis=1, inplace=True)
    print("")
    print(tabulate(module_results_df, headers="keys"))
