# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""examples/trl_async_grpo.py end to end on CPU: TRL's AsyncGRPOTrainer builds and
trains a tiny Qwen3.5 on tree rows, fed by a stub rollout worker, with no weight sync
and a PyTorch stand-in for TRL's FA3 tree attention."""

import importlib.util
import itertools
import queue
from pathlib import Path

import numpy as np
import pytest
import torch
from test_gated_delta_net import qwen3_5
from tree_recurrence.integrations import transformers as integration
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

EXAMPLE = Path(__file__).parents[1] / "examples" / "trl_async_grpo.py"
PROMPT = 6


def example():
    spec = importlib.util.spec_from_file_location("trl_async_grpo", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def build_trainer(tmp_path):
    model_dir = tmp_path / "model"
    qwen3_5().save_pretrained(model_dir)
    return AsyncGRPOTrainer(
        model=str(model_dir),
        args=config(tmp_path),
        train_dataset=Dataset.from_dict(
            {"prompt": [[{"role": "user", "content": ""}]] * 8}
        ),
        processing_class=tokenizer(),
        rollout_worker=Rollouts(branching_sequences(vocab_size=62)),
        weight_transfer=NoWeightSync(),
    )


def test_trl_refuses_the_model_by_itself(tmp_path, cpu_trl):
    with pytest.raises(ValueError, match="linear_attention"):
        build_trainer(tmp_path)


def test_async_grpo_trains_a_hybrid_model_on_tree_rows(tmp_path, monkeypatch, cpu_trl):
    trees = []
    tree_scan = integration.tree_scan

    def spy(scan, inputs, plan, **kwargs):
        trees.append(plan.num_levels)
        return tree_scan(scan, inputs, plan, **kwargs)

    monkeypatch.setattr(integration, "tree_scan", spy)

    with example().tree_recurrent_models() as supports:
        trainer = build_trainer(tmp_path)
    assert [s.owners for s in supports] == [{"Qwen3_5GatedDeltaNet": "tree_recurrence"}]
    # hidden from TRL's check only while it built the trainer
    assert trainer.model.config.get_text_config().layer_types[0] == "linear_attention"
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    trainer.train()

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    assert len(losses) == 2 and all(np.isfinite(losses))
    # the GDN layers ran on rows whose samples shared prefixes
    assert trees and max(trees) > 1
    moved = [
        n
        for n, p in trainer.model.named_parameters()
        if "linear_attn" in n and not torch.equal(p, before[n])
    ]
    assert moved
