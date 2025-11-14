"""
E2E comparison test for GLM4 MoE grouped kernels.
"""

from __future__ import annotations

import copy
import os
import random
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from functools import wraps

import numpy as np
import torch
from datasets import Dataset

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.train import train
from axolotl.loaders import load_tokenizer
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


def with_temp_dir(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        temp_dir = tempfile.mkdtemp()
        try:
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            shutil.rmtree(temp_dir)

    return wrapper


class TestGlm4MoeGrouped(unittest.TestCase):
    """Ensure grouped kernels behave identically to vanilla GLM4 MoE."""

    _BASE_CFG = {
        "base_model": "tiny-random/glm-4-moe",
        "tokenizer_config": "tiny-random/glm-4-moe",
        "trust_remote_code": True,
        "flash_attention": False,
        "sequence_len": 256,
        "bf16": False,
        "fp16": False,
        "val_set_size": 0.0,
        "special_tokens": {},
        "datasets": [
            {
                "path": "axolotl/tests/fixtures/alpaca/alpaca.json",
                "type": "alpaca",
            },
        ],
        "num_epochs": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "max_steps": 2,
        "save_steps": 0,
        "eval_steps": 0,
        "save_first_step": False,
        "logging_steps": 1,
        "report_to": [],
        "gradient_checkpointing": False,
        "train_on_inputs": False,
        "seed": 1234,
        "skip_prepare_dataset": False,
        "remove_unused_columns": False,
        "dataset_num_proc": 1,
    }

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_training(
        self,
        temp_dir: str,
        mlp_impl: str,
        *,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> tuple[int, list[tuple[int, float]]]:
        cfg_dict = copy.deepcopy(self._BASE_CFG)
        cfg_dict["output_dir"] = os.path.join(temp_dir, mlp_impl)
        cfg_dict["bf16"] = False
        cfg_dict["fp16"] = False

        if mlp_impl == "grouped":
            cfg_dict["mlp_impl"] = "grouped"
            cfg_dict["use_grouped_moe_kernels"] = True
        elif mlp_impl == "megablocks":
            cfg_dict["mlp_impl"] = "megablocks"
            cfg_dict["use_grouped_moe_kernels"] = True
            cfg_dict["bf16"] = True
        else:
            cfg_dict["use_grouped_moe_kernels"] = False
            cfg_dict.pop("mlp_impl", None)

        if max_steps is not None:
            cfg_dict["max_steps"] = max_steps
        if seed is not None:
            cfg_dict["seed"] = seed

        cache_dir = os.path.join(temp_dir, "hf-cache")
        os.makedirs(cache_dir, exist_ok=True)
        prev_cache = os.environ.get("HF_DATASETS_CACHE")
        os.environ["HF_DATASETS_CACHE"] = cache_dir

        from datasets import config as datasets_config  # type: ignore

        prev_cache_cfg = datasets_config.HF_DATASETS_CACHE
        datasets_config.HF_DATASETS_CACHE = cache_dir

        try:
            cfg = DictDefault(cfg_dict)
            cfg = validate_config(cfg)
            normalize_config(cfg)
            prepare_plugins(cfg)

            tokenizer = load_tokenizer(cfg)
            prompts = [
                "Reverse the token order: Hello world",
                "Summarize input: Axolotl enables fast fine-tuning.",
            ]
            encoded = tokenizer(
                prompts,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=True,
            )
            dataset_dict = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": [ids[:] for ids in encoded["input_ids"]],
            }
            train_dataset = Dataset.from_dict(dataset_dict)
            dataset_meta = TrainDatasetMeta(train_dataset=train_dataset)

            self._set_seed(int(cfg.seed or 0))

            with nullcontext():
                model, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
        finally:
            if prev_cache is None:
                os.environ.pop("HF_DATASETS_CACHE", None)
            else:
                os.environ["HF_DATASETS_CACHE"] = prev_cache
            datasets_config.HF_DATASETS_CACHE = prev_cache_cfg

        patched_count = sum(
            1 for module in model.modules() if getattr(module, "_axolotl_grouped_moe", False)
        )
        loss_entries: list[tuple[int, float]] = []
        for entry in trainer.state.log_history:
            if not isinstance(entry, dict) or "loss" not in entry:
                continue
            step = int(entry.get("step", len(loss_entries) + 1))
            loss_entries.append((step, float(entry["loss"])))

        self.assertGreater(len(loss_entries), 0, "Expected training loss history to be populated.")

        if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "free_memory"):
            trainer.accelerator.free_memory()

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return patched_count, loss_entries

    @with_temp_dir
    def test_grouped_matches_vanilla_training(self, temp_dir: str):
        vanilla_patched, vanilla_trace = self._run_training(temp_dir, mlp_impl="vanilla")
        grouped_patched, grouped_trace = self._run_training(temp_dir, mlp_impl="grouped")

        self.assertEqual(vanilla_patched, 0, "Vanilla run should not apply grouped kernels.")
        self.assertGreater(grouped_patched, 0, "Grouped run did not patch any MoE blocks.")
        self.assertEqual(
            len(vanilla_trace),
            len(grouped_trace),
            "Loss histories should have identical lengths.",
        )

        for (step_v, vanilla), (step_g, grouped) in zip(vanilla_trace, grouped_trace):
            self.assertEqual(step_v, step_g, "Logged training steps should align between runs.")
            self.assertAlmostEqual(
                vanilla,
                grouped,
                delta=1e-4,
                msg=f"Loss diverged at step {step_v}: vanilla={vanilla}, grouped={grouped}",
            )

    @with_temp_dir
    def test_megablocks_matches_vanilla(self, temp_dir: str):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm device required for MegaBlocks parity test.")

        try:
            from axolotl.monkeypatch.models.bailing_moe_v2.modeling import (
                _load_megablocks_backend,
            )
        except ImportError:
            self.skipTest("MegaBlocks backend loader unavailable; skipping.")

        if _load_megablocks_backend() is None:
            self.skipTest("MegaBlocks backend not available; skipping parity test.")

        vanilla_patched, vanilla_trace = self._run_training(temp_dir, mlp_impl="vanilla")
        megablocks_patched, megablocks_trace = self._run_training(
            temp_dir,
            mlp_impl="megablocks",
        )

        self.assertEqual(vanilla_patched, 0, "Vanilla run should not apply grouped kernels.")
        self.assertGreater(megablocks_patched, 0, "MegaBlocks run did not patch any MoE blocks.")
        self.assertEqual(
            len(vanilla_trace),
            len(megablocks_trace),
            "Loss histories should have identical lengths.",
        )

        for (step_v, vanilla), (step_m, megablocks) in zip(vanilla_trace, megablocks_trace):
            self.assertEqual(step_v, step_m, "Logged training steps should align between runs.")
            self.assertAlmostEqual(
                vanilla,
                megablocks,
                delta=1e-4,
                msg=f"Loss diverged at step {step_v}: vanilla={vanilla}, megablocks={megablocks}",
            )
