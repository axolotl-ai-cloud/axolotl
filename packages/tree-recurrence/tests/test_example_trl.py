# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The TRL examples' setup end to end on CPU: under ``tree_recurrent_models``, TRL's
AsyncGRPOTrainer builds and trains tiny Qwen3.5, Nemotron-H and LFM2-MoE models (with
the recipes' LoRA) on tree rows, fed by a stub rollout worker, with no weight sync and
a PyTorch stand-in for TRL's FA3 tree attention."""

import importlib.util
import itertools
import queue
from pathlib import Path

import numpy as np
import pytest
import torch
from test_gated_delta_net import qwen3_5
from test_lfm2 import build as build_lfm2
from test_mamba2 import build as build_mamba2
from tree_recurrence.components import short_conv
from tree_recurrence.integrations import (
    transformers as integration,
    trl as trl_integration,
)
from tree_testing import branching_sequences, reference_tree_attention

pytest.importorskip("trl.experimental.async_grpo")
from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from trl.experimental.async_grpo import (  # noqa: E402
    AsyncGRPOConfig,
    AsyncGRPOTrainer,
    async_grpo_trainer,
    packing,
)
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample  # noqa: E402

EXAMPLES = Path(__file__).parents[1] / "examples"
PROMPT = 6


def example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nemotron_h():
    return build_mamba2("nemotron_h")[1]


def lfm2_moe():
    return build_lfm2("lfm2_moe")[1]


def recipe_lora(name):
    def lora():
        from peft import LoraConfig

        return LoraConfig(r=4, lora_alpha=8, target_modules=example(name).LORA_TARGETS)

    return lora


# model, its recurrent mixer, a parameter name every step must change, LoRA config
MODELS = {
    "qwen3_5": (qwen3_5, "Qwen3_5GatedDeltaNet", "linear_attn", None),
    "nemotron_h": (nemotron_h, "NemotronHMamba2Mixer", "mixer.in_proj", None),
    "nemotron_h-lora": (
        nemotron_h,
        "NemotronHMamba2Mixer",
        "in_proj.lora_B",
        recipe_lora("nemotron_3_5_lightning"),
    ),
    "lfm2_moe-lora": (
        lfm2_moe,
        "Lfm2MoeShortConv",
        "conv.in_proj.lora_B",
        recipe_lora("lfm2_5_8b_a1b"),
    ),
}
MOE = {"nemotron_h", "nemotron_h-lora", "lfm2_moe-lora"}


class Rollouts:
    """A rollout worker replaying groups of samples that share their prompts, tagged
    with the live policy version and refilled at every weight sync."""

    def __init__(self, sequences, per_sync=48):
        self.rollout_buffer: queue.Queue = queue.Queue()
        self.metrics_queue: queue.Queue = queue.Queue()
        self.model_version = 0
        self.per_sync = per_sync
        self.rng = np.random.default_rng(0)
        self.groups = itertools.count()
        self.rows = itertools.cycle(enumerate(sequences))

    def _fill(self):
        for _ in range(self.per_sync):
            i, ids = next(self.rows)
            if i % 4 == 0:
                group = next(self.groups)
            prompt = min(PROMPT, len(ids) - 1)
            self.rollout_buffer.put(
                RolloutSample(
                    prompt=[{"role": "user", "content": ""}],
                    completion=[{"role": "assistant", "content": ""}],
                    input_ids=np.asarray(ids, dtype=np.int64),
                    completion_mask=np.asarray(
                        [0] * prompt + [1] * (len(ids) - prompt), dtype=np.int64
                    ),
                    old_log_probs=np.full(len(ids), -4.0, dtype=np.float32),
                    advantage=float(self.rng.normal()),
                    model_version=self.model_version,
                    group_id=group,
                    metrics={},
                )
            )

    def start(self):
        self._fill()

    def update_model_version(self, model_version):
        self.model_version = model_version
        self._fill()

    def stop(self):
        pass

    def check_health(self, stale_after_s):
        pass


class NoWeightSync:
    def init_weight_transfer(self):
        pass

    def pause(self):
        pass

    def send_weights(self, iterator):
        for _ in iterator:
            pass

    def resume(self):
        pass

    def destroy(self):
        pass


def tokenizer():
    vocab = {str(i): i for i in range(62)} | {"<pad>": 62, "<eos>": 63}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="0")),
        pad_token="<pad>",
        eos_token="<eos>",
    )


@pytest.fixture
def cpu_trl(monkeypatch):
    """TRL's trainer without a vLLM server, a CUDA device, or FA3."""
    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    monkeypatch.setattr(async_grpo_trainer, "VLLMClient", lambda *a, **k: None)
    # its MFU metric reads the CUDA device's name
    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a: None
    )


def config(tmp_path):
    return AsyncGRPOConfig(
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=8,
        num_generations=4,
        token_budget=128,
        learning_rate=1e-3,
        use_cpu=True,
        bf16=False,
        dtype="float32",
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )


def build_trainer(tmp_path, model=qwen3_5, peft_config=None):
    model_dir = tmp_path / "model"
    model().save_pretrained(model_dir)
    return AsyncGRPOTrainer(
        model=str(model_dir),
        args=config(tmp_path),
        train_dataset=Dataset.from_dict(
            {"prompt": [[{"role": "user", "content": ""}]] * 8}
        ),
        processing_class=tokenizer(),
        rollout_worker=Rollouts(branching_sequences(vocab_size=62)),
        weight_transfer=NoWeightSync(),
        peft_config=peft_config,
    )


def test_trl_refuses_the_model_by_itself(tmp_path, cpu_trl):
    with pytest.raises(ValueError, match="linear_attention"):
        build_trainer(tmp_path)


@pytest.mark.parametrize("name", list(MODELS))
def test_async_grpo_trains_a_hybrid_model_on_tree_rows(
    tmp_path, monkeypatch, cpu_trl, name
):
    model, mixer, trained, peft_config = MODELS[name]
    if name in MOE:
        guarded = trl_integration.guarded_flops
        flops = async_grpo_trainer.compute_flops_per_token
        monkeypatch.setattr(
            async_grpo_trainer, "compute_flops_per_token", guarded(flops)
        )
    if peft_config is not None:
        pytest.importorskip("peft")
        peft_config = peft_config()
    trees = []

    def spy(owner, name):
        run = getattr(owner, name)

        def call(fn, x, plan, *args, **kwargs):
            trees.append(plan.num_levels)
            return run(fn, x, plan, *args, **kwargs)

        monkeypatch.setattr(owner, name, call)

    # the recurrences' scans, and LFM2's convs (it has no scan)
    spy(integration, "tree_scan")
    spy(short_conv, "tree_conv")

    with trl_integration.tree_recurrent_models() as supports:
        trainer = build_trainer(tmp_path, model, peft_config)
    assert [s.owners for s in supports] == [{mixer: "tree_recurrence"}]
    # hidden from TRL's check only while it built the trainer
    layer_types = trainer.model.config.get_text_config().layer_types
    assert set(layer_types) & {"linear_attention", "conv"}
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    trainer.train()

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    assert len(losses) == 2 and all(np.isfinite(losses))
    # the recurrent layers ran on rows whose samples shared prefixes
    assert trees and max(trees) > 1
    moved = [
        n
        for n, p in trainer.model.named_parameters()
        if trained in n and not torch.equal(p, before[n])
    ]
    assert moved
