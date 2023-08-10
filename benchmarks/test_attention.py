"""Benchmarks for various attention mechanisms"""

import gc

import pytest
import torch
from configs import TestConfigs
from pytest_cases import parametrize_with_cases
from tabulate import tabulate  # type: ignore

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.models import load_model, load_tokenizer


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
def test_load_model(model_cfg, dtype_cfg, results_bag):
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
        [
            "status",
            "model_cfg",
            "attn_cfg",
            "dtype_cfg",
            "adapter_cfg",
            "pytest_obj",
            "vram_baseline",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    print("")
    print(tabulate(module_results_df, headers="keys"))
