"""Benchmarks for various attention mechanisms"""

import gc

import pytest
import torch
from pytest_cases import parametrize_with_cases
from tabulate import tabulate  # type: ignore

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer


class TestConfigs:  # pylint: disable=missing-class-docstring
    def model_llama2(self):
        return DictDefault(
            {
                "base_model": "meta-llama/Llama-2-7b-chat-hf",
                "base_model_config": "meta-llama/Llama-2-7b-chat-hf",
                "model_type": "LlamaForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "sequence_len": 1024,
                "gradient_accumulation_steps": 1,
                "micro_batch_size": 1,
                "pad_token": "<pad>",
            }
        )

    def attn_base(self):
        return DictDefault({})

    def attn_xformers(self):
        return DictDefault(
            {
                "xformers_attention": True,
            }
        )

    def attn_flash(self):
        return DictDefault(
            {
                "flash_attention": True,
            }
        )

    def lora_params(self):
        return DictDefault(
            {
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_linear": True,
            }
        )

    def adapter_none(self):
        return DictDefault({})

    def adapter_lora(self):
        return (
            DictDefault(
                {
                    "adapter": "lora",
                }
            )
            | self.lora_params()
        )

    def adapter_qlora(self):
        return (
            DictDefault(
                {
                    "adapter": "qlora",
                }
            )
            | self.lora_params()
        )

    def dtype_fp32(self):
        return DictDefault(
            {
                "fp32": True,
            }
        )

    def dtype_bf16(self):
        return DictDefault(
            {
                "bf16": True,
            }
        )

    def dtype_4bit(self):
        return (
            DictDefault(
                {
                    "load_in_4bit": True,
                }
            )
            | self.adapter_qlora()
        )

    def dtype_8bit(self):
        return (
            DictDefault(
                {
                    "load_in_8bit": True,
                }
            )
            | self.adapter_lora()
        )


@pytest.fixture(autouse=True)
def memory_cleanup():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("attn_cfg", cases=TestConfigs, prefix="attn_")
def test_benchmark_attn(model_cfg, attn_cfg, results_bag):
    cfg = model_cfg | attn_cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)
    model = load_model(cfg, tokenizer)
    for k, val in cfg.stats_bag.items():
        results_bag[k] = val
    del tokenizer
    del model


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
def test_bench_load_model(model_cfg, dtype_cfg, results_bag):
    cfg = model_cfg | dtype_cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)
    model = load_model(cfg, tokenizer)
    for k, val in cfg.stats_bag.items():
        results_bag[k] = val
    del tokenizer
    del model


def test_synthesis(module_results_df):
    module_results_df.drop(
        ["status", "model_cfg", "attn_cfg", "dtype_cfg", "adapter_cfg", "pytest_obj"],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    print("")
    print(tabulate(module_results_df, headers="keys"))
