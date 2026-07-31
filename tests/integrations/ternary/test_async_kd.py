"""Async KD rollout components: worker generation, staleness, masking."""

import queue
import time

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.async_kd import (
    IGNORE_INDEX,
    HFGenerateRolloutWorker,
    KDRolloutCollator,
    KDRolloutDataset,
    KDSample,
)


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )


def _drain(worker, n, timeout=30.0):
    out = []
    deadline = time.monotonic() + timeout
    while len(out) < n and time.monotonic() < deadline:
        try:
            out.append(worker.rollout_buffer.get(timeout=1.0))
        except queue.Empty:
            worker.check_health(stale_after_s=timeout)
    assert len(out) == n, f"drained {len(out)}/{n}"
    return out


def test_worker_generates_prompt_prefixed_samples():
    worker = HFGenerateRolloutWorker(
        model=_tiny_model(),
        prompts=[[5, 6, 7], [8, 9, 10, 11]],
        pad_token_id=0,
        max_new_tokens=6,
        batch_size=2,
        buffer_size=8,
    )
    worker.start()
    try:
        samples = _drain(worker, 4)
    finally:
        worker.stop()
    for sample in samples:
        assert sample.input_ids[: sample.prompt_len] in ([5, 6, 7], [8, 9, 10, 11])
        assert 0 < len(sample.input_ids) - sample.prompt_len <= 6
        assert sample.model_version == 0


def test_worker_reports_version_after_sync():
    model = _tiny_model()
    worker = HFGenerateRolloutWorker(
        model=model,
        prompts=[[3, 4]],
        pad_token_id=0,
        max_new_tokens=4,
        batch_size=1,
        buffer_size=2,
    )
    worker.sync_weights(model.state_dict(), version=7)
    worker.start()
    try:
        samples = _drain(worker, 2)
    finally:
        worker.stop()
    assert all(s.model_version == 7 for s in samples)


def test_worker_health_surfaces_thread_failure():
    worker = HFGenerateRolloutWorker(
        model=_tiny_model(),
        prompts=[[999999]],  # out-of-vocab -> embedding index error in the thread
        pad_token_id=0,
        batch_size=1,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 15
        with pytest.raises(RuntimeError, match="async KD worker"):
            while time.monotonic() < deadline:
                worker.check_health(stale_after_s=15)
                time.sleep(0.2)
    finally:
        worker.stop()


def test_empty_prompt_pool_refused():
    with pytest.raises(ValueError, match="prompt pool"):
        HFGenerateRolloutWorker(model=_tiny_model(), prompts=[], pad_token_id=0)


def test_dataset_masks_prompts_and_drops_stale():
    buffer = queue.Queue()
    buffer.put(KDSample(input_ids=[1, 2, 3, 4, 5], prompt_len=2, model_version=0))
    buffer.put(KDSample(input_ids=[6, 7, 8], prompt_len=1, model_version=9))
    dataset = KDRolloutDataset(
        buffer, version_fn=lambda: 10, max_staleness=4, max_samples=1
    )

    rows = list(dataset)
    assert len(rows) == 1
    assert dataset.dropped_stale == 1
    assert rows[0]["input_ids"] == [6, 7, 8]
    assert rows[0]["labels"] == [IGNORE_INDEX, 7, 8]


def test_collator_pads_and_masks():
    batch = KDRolloutCollator(pad_token_id=0)(
        [
            {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]},
            {"input_ids": [4, 5], "labels": [IGNORE_INDEX, 5]},
        ]
    )
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [4, 5, 0]
    assert batch["labels"][1].tolist() == [IGNORE_INDEX, 5, IGNORE_INDEX]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
