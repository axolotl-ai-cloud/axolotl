"""Benchmarks for various attention mechanisms"""

import functools
import gc
import logging
from pathlib import Path

import pytest
import torch
from configs import TestConfigs
from datasets import Dataset
from pytest_cases import parametrize_with_cases
from tabulate import tabulate  # type: ignore

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.data import encode_pretraining
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.wandb import setup_wandb_env_vars

logs_dir = Path(__file__).parent / "logs"


@pytest.fixture(autouse=True)
def configure_logging(request, caplog):
    log_file = logs_dir / f"{request.node.name}.log"
    request.config.pluginmanager.get_plugin("logging-plugin").set_log_path(log_file)
    caplog.set_level(logging.DEBUG)


@pytest.fixture(autouse=True)
def memory_cleanup(results_bag):
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if "cfg" in results_bag:
            for k, val in results_bag.cfg.stats_bag.items():
                results_bag[k] = val


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("attn_cfg", cases=TestConfigs, prefix="attn_")
def test_benchmark_attn(model_cfg, attn_cfg, results_bag):
    cfg = model_cfg | TestConfigs().dtype_bf16() | attn_cfg
    cfg.output_dir = logs_dir
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    tokenizer = load_tokenizer(cfg)
    model, _ = load_model(cfg, tokenizer)

    dataset = Dataset.from_list([{"text": "hello world"}])
    encode = functools.partial(encode_pretraining, tokenizer, cfg.sequence_len)
    dataset = dataset.map(encode, batched=True, remove_columns=["text"])
    trainer = setup_trainer(cfg, dataset.with_format("torch"), [], model, tokenizer)
    trainer.train()

    del trainer
    del tokenizer
    del model


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
def _test_load_model(model_cfg, dtype_cfg, results_bag):
    cfg = model_cfg | dtype_cfg
    cfg.output_dir = logs_dir
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    assert cfg.stats_bag.vram_baseline <= 1.750
    tokenizer = load_tokenizer(cfg)
    model, _ = load_model(cfg, tokenizer)
    del tokenizer
    del model


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases(
    "dtype_cfg", cases=TestConfigs, prefix="dtype_", has_tag="quick"
)
def _test_trainer(model_cfg, dtype_cfg, results_bag):
    cfg = model_cfg | dtype_cfg
    cfg.output_dir = logs_dir
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    assert cfg.stats_bag.vram_baseline <= 3
    tokenizer = load_tokenizer(cfg)
    model, _ = load_model(cfg, tokenizer)

    dataset = Dataset.from_list([{"text": "hello world"}])
    encode = functools.partial(encode_pretraining, tokenizer, cfg.sequence_len)
    dataset = dataset.map(encode, batched=True, remove_columns=["text"])
    trainer = setup_trainer(cfg, dataset.with_format("torch"), [], model, tokenizer)
    trainer.train()

    del trainer
    del tokenizer
    del model


def test_synthesis(module_results_df):
    module_results_df.drop(
        [
            "status",
            "cfg",
            "model_cfg",
            "attn_cfg",
            "dtype_cfg",
            "adapter_cfg",
            "pytest_obj",
            "vram_baseline",
            "vram_last",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    print("")
    print(tabulate(module_results_df, headers="keys"))
