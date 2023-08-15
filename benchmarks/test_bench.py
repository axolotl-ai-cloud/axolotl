"""Benchmarks for various attention mechanisms"""

import functools
import gc
import json
import logging
import time
from pathlib import Path

import pytest
import torch
from configs import TestConfigs
from datasets import Dataset
from pytest_cases import parametrize_with_cases
from pytorch_memlab import MemReporter
from transformers import GenerationConfig

from axolotl.utils.bench import (
    gpu_memory_usage,
    gpu_memory_usage_all,
    log_gpu_memory_usage,
)
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.data import encode_pretraining
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.wandb import setup_wandb_env_vars

LOG = logging.getLogger("axolotl.bench")
logs_dir = Path(__file__).parent / "logs"


@pytest.fixture(autouse=True)
def capture_logs(request, caplog):
    log_file = logs_dir / f"{request.node.name}.log"
    request.config.pluginmanager.get_plugin("logging-plugin").set_log_path(log_file)
    caplog.set_level(logging.DEBUG)


@pytest.fixture(autouse=True)
def copy_results(results_bag):
    try:
        yield
    finally:
        if (
            cfg := results_bag.pop("cfg", None)
        ) is not None and cfg.stats_bag is not None:
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
                value = value.copy()
                value["test_id"] = name
                file.write(json.dumps(value) + "\n")


@pytest.fixture(autouse=True)
def memory_cleanup():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        yield
    except torch.cuda.OutOfMemoryError:
        MemReporter().report()
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()

        if (mem := gpu_memory_usage()) > 3.0:
            LOG.warning("GPU memory usage still high!")
            cnt = 0
            for obj in get_tensors():
                obj.detach()
                obj.grad = None
                obj.storage().resize_(0)
                cnt += 1
            gc.collect()
            torch.cuda.empty_cache()
            usage, cache, misc = gpu_memory_usage_all()
            LOG.warning(
                f"  forcibly cleared {cnt} tensors: {mem:.03f}GB -> {usage:.03f}GB (+{cache:.03f}GB cache, +{misc:.03f}GB misc)"
            )


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
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue


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
    assert cfg.stats_bag.vram_baseline <= 0.25
    tokenizer = load_tokenizer(cfg)
    model, _ = load_model(cfg, tokenizer)
    del tokenizer
    del model


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
@parametrize_with_cases("attn_cfg", cases=TestConfigs, prefix="attn_")
@parametrize_with_cases("ctx_len", cases=TestConfigs, prefix="ctx_")
def test_inference(model_cfg, dtype_cfg, attn_cfg, ctx_len, results_bag):
    cfg = model_cfg | dtype_cfg | attn_cfg
    cfg.output_dir = str(logs_dir.resolve())
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    try:
        validate_config(cfg)
    except ValueError as ex:
        pytest.skip(str(ex))
        return
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    assert cfg.stats_bag.vram_baseline <= 0.25
    try:
        tokenizer = load_tokenizer(cfg)
        model, _ = load_model(cfg, tokenizer)
        batch = tokenizer(
            "hello world " * int(ctx_len / 2),
            return_tensors="pt",
            add_special_tokens=False,
        )
        cfg.stats_bag.prompt_tokens = batch["input_ids"].shape[1]
        model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                min_new_tokens=128,
                max_new_tokens=128,
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            start = time.time()
            torch.cuda.empty_cache()
            generated = model.generate(
                inputs=batch["input_ids"].to(model.device),
                generation_config=generation_config,
                # streamer=TextStreamer(tokenizer),
            )
            _, cache, _ = log_gpu_memory_usage(LOG, "after inference", model.device)
            cfg.stats_bag.vram_generate_cache = cache
            cfg.stats_bag.generate_time = time.time() - start
            cfg.stats_bag.generate_tokens = (
                generated["sequences"].shape[1] - cfg.stats_bag.prompt_tokens
            )
            cfg.stats_bag.generate_tps = (
                cfg.stats_bag.generate_tokens / cfg.stats_bag.generate_time
            )
            LOG.debug(
                tokenizer.decode(
                    generated["sequences"]
                    .cpu()
                    .tolist()[0][cfg.stats_bag.prompt_tokens :]
                )
            )
    finally:
        try:
            del tokenizer
            del model
        except UnboundLocalError:
            pass


@parametrize_with_cases("model_cfg", cases=TestConfigs, prefix="model_")
@parametrize_with_cases("attn_cfg", cases=TestConfigs, prefix="attn_")
@parametrize_with_cases("dtype_cfg", cases=TestConfigs, prefix="dtype_")
@parametrize_with_cases("ctx_len", cases=TestConfigs, prefix="ctx_")
# @parametrize_with_cases("opt_cfg", cases=TestConfigs, prefix="opt_")
@parametrize_with_cases(
    "opt_cfg", cases=TestConfigs, prefix="opt_", glob="opt_adamw_bnb_8bit"
)
def test_trainer(model_cfg, attn_cfg, dtype_cfg, opt_cfg, ctx_len, results_bag):
    cfg = (
        model_cfg
        | opt_cfg
        | dtype_cfg
        | attn_cfg
        | DictDefault({"sequence_len": ctx_len})
    )
    cfg.output_dir = str(logs_dir.resolve())
    results_bag.cfg = cfg
    assert "llama" in cfg.base_model
    assert validate_config(cfg) is None
    normalize_config(cfg)
    setup_wandb_env_vars(cfg)
    assert cfg.stats_bag.vram_baseline <= 0.25
    try:
        trainer = None
        tokenizer = load_tokenizer(cfg)
        model, _ = load_model(cfg, tokenizer)

        dataset = Dataset.from_list(
            [{"text": "hello world " * int(ctx_len / 2)} for _ in range(10)]
        )
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
