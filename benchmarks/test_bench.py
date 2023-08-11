"""Benchmarks for various attention mechanisms"""

import functools
import gc
import json
import logging
from pathlib import Path

import pytest
import torch
from configs import TestConfigs
from datasets import Dataset
from pytest_cases import parametrize_with_cases

from axolotl.utils.bench import gpu_memory_usage
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
def copy_results(results_bag):
    try:
        yield
    finally:
        if (cfg := results_bag.pop("cfg", None)) is not None:
            for key, val in cfg.stats_bag.items():
                results_bag[key] = val


@pytest.fixture(scope="session", autouse=True)
def write_json(fixture_store):
    try:
        yield
    finally:
        out = fixture_store["results_bag"]
        for key, value in out.items():
            name = key.split("::")[1]
            with open(logs_dir / f"{name}.jsonl", "w", encoding="UTF-8") as file:
                file.write(json.dumps(value) + "\n")


@pytest.fixture(autouse=True)
def memory_cleanup():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()

        if (mem := gpu_memory_usage()) > 3.0:
            print("GPU memory usage still high!")
            cnt = 0
            for obj in get_tensors():
                obj.detach()
                obj.grad = None
                obj.storage().resize_(0)
                cnt += 1
            print(f"  forcibly cleared {cnt} tensors using {mem} GB")
            gc.collect()
            torch.cuda.empty_cache()


def get_tensors(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda or not gpu_only:
                yield tensor
        except (
            RuntimeError,
            ModuleNotFoundError,
            OSError,
            AssertionError,
            ImportError,
        ):
            pass


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("attn_cfg", cases=TestConfigs, prefix="attn_")
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
def test_bench_attn(model_cfg, attn_cfg, dtype_cfg, results_bag):
    cfg = model_cfg | dtype_cfg | attn_cfg
    cfg.output_dir = str(logs_dir.resolve())
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    assert cfg.stats_bag.vram_baseline <= 3
    try:
        trainer = None
        tokenizer = load_tokenizer(cfg)
        model, _ = load_model(cfg, tokenizer)

        dataset = Dataset.from_list([{"text": "hello world"}])
        encode = functools.partial(encode_pretraining, tokenizer, cfg.sequence_len)
        dataset = dataset.map(encode, batched=True, remove_columns=["text"])

        trainer = setup_trainer(cfg, dataset.with_format("torch"), [], model, tokenizer)
        trainer.train()
        for elem in trainer.state.log_history:
            if "train_runtime" in elem:
                cfg.stats_bag["train_result"] = elem
                for key, val in elem.items():
                    if key == "train_runtime":
                        key = "time_train"
                    elif key == "train_samples_per_second":
                        ...
                    else:
                        continue
                    results_bag[key] = val

    finally:
        if trainer is not None:
            opt = trainer.optimizer
            trainer.optimizer = None
            opt.zero_grad()
            del opt
            del trainer
        del tokenizer
        del model


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
def test_load_model(model_cfg, dtype_cfg, results_bag):
    cfg = model_cfg | dtype_cfg
    cfg.output_dir = str(logs_dir.resolve())
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
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
@parametrize_with_cases("opt_cfg", cases=TestConfigs, prefix="opt_")
def test_trainer(model_cfg, opt_cfg, dtype_cfg, results_bag):
    cfg = model_cfg | opt_cfg | dtype_cfg
    cfg.output_dir = str(logs_dir.resolve())
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    assert cfg.stats_bag.vram_baseline <= 3
    try:
        trainer = None
        tokenizer = load_tokenizer(cfg)
        model, _ = load_model(cfg, tokenizer)

        dataset = Dataset.from_list([{"text": "hello world"}])
        encode = functools.partial(encode_pretraining, tokenizer, cfg.sequence_len)
        dataset = dataset.map(encode, batched=True, remove_columns=["text"])

        trainer = setup_trainer(cfg, dataset.with_format("torch"), [], model, tokenizer)
        trainer.train()
        for elem in trainer.state.log_history:
            if "train_runtime" in elem:
                cfg.stats_bag["train_result"] = elem
                for key, val in elem.items():
                    if key == "train_runtime":
                        key = "time_train"
                    elif key == "train_samples_per_second":
                        ...
                    else:
                        continue
                    results_bag[key] = val

    finally:
        if trainer is not None:
            del trainer
        del tokenizer
        del model
