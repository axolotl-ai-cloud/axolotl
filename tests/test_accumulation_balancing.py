"""Accumulation grouping follows the rank-strided microbatch schedule."""

import numpy as np
import pytest
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from axolotl.utils.samplers import LabelBalancedRandomSampler, MultipackBatchSampler
from axolotl.utils.samplers.accumulation import (
    accumulation_metrics,
    balance_accumulation,
)


@pytest.mark.parametrize("replicas", [1, 2, 4])
@pytest.mark.parametrize("steps", [1, 2, 4, 17])
def test_accumulation_objectives_and_tail(replicas, steps):
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 10000, size=steps * replicas * 19 + 3).tolist()
    batches = [[[i]] for i in range(len(labels))]
    after = balance_accumulation(batches, labels, steps * replicas)
    reordered = [labels[b[0][0]] for b in after]
    assert sorted(b[0][0] for b in after) == list(range(len(labels)))
    full = len(labels) // (steps * replicas) * (steps * replicas)
    assert after[full:] == batches[full:]
    if steps * replicas == 1:
        assert after == batches
    a, b = (
        accumulation_metrics(labels, steps * replicas),
        accumulation_metrics(reordered, steps * replicas),
    )
    assert b["std_global_update_labels"] <= a["std_global_update_labels"] + 1e-8
    assert b["mean_global_update_labels"] == a["mean_global_update_labels"]
    assert after == balance_accumulation(batches, labels, steps * replicas)


def test_concentrated_windows_are_smoothed():
    labels = [9] * 16 + [1] * 16
    before = [[[i]] for i in range(32)]
    after = balance_accumulation(before, labels, 8)
    totals = accumulation_metrics([labels[b[0][0]] for b in after], 8)
    assert totals["std_global_update_labels"] == 0


@pytest.mark.parametrize("kind", ["packed", "flattened", "padded"])
@pytest.mark.parametrize("replicas, steps", [(1, 4), (2, 1), (2, 4), (4, 1), (4, 4)])
def test_sampler_integration_matches_actual_rank_shards(kind, replicas, steps):
    lengths = np.full(523, 4)
    counts = np.tile([0, 1, 2, 3], 131)[:523]
    if kind == "packed":
        sampler = MultipackBatchSampler(
            list(range(523)),
            lengths=lengths,
            label_counts=counts,
            batch_size=1,
            batch_max_len=16,
            bin_size=4,
            num_processes=1,
            seed=42,
            batches_per_optimizer_step=replicas * steps,
        )
        batches = sampler.generate_batches()
        sampler._len_across_ranks = len(batches)
        batch_sampler = sampler
        label_count = lambda batch: sum(int(counts[i]) for row in batch for i in row)
    else:
        sampler = LabelBalancedRandomSampler(
            lengths,
            counts,
            4,
            length_mode=kind,
            seed=42,
            batches_per_optimizer_step=replicas * steps,
        )
        batch_sampler = BatchSampler(sampler, 4, True)
        list(sampler)
        label_count = lambda batch: sum(int(counts[i]) for i in batch)
    shards = [
        list(
            BatchSamplerShard(
                batch_sampler,
                num_processes=replicas,
                process_index=r,
                even_batches=False,
            )
        )
        for r in range(replicas)
    ]
    n = min(len(s) for s in shards) // steps * steps
    totals = np.array(
        [
            [
                sum(label_count(b) for b in shard[i : i + steps])
                for i in range(0, n, steps)
            ]
            for shard in shards
        ]
    )
    metrics = sampler.label_metrics
    assert metrics["after"]["std_global_update_labels"] == pytest.approx(
        totals.sum(axis=0).std()
    )
    for key in ["std_global_update_labels"]:
        assert metrics["after"][key] <= metrics["before_accumulation"][key] + 1e-8
    for key in ["mean_label_count", "std_label_count", "total_label_count"]:
        assert metrics["after"][key] == pytest.approx(
            metrics["before_accumulation"][key]
        )


def test_incomplete_update_unchanged():
    batches = [[i] for i in range(7)]
    assert balance_accumulation(batches, list(range(7)), 8) == batches


def test_global_objective_does_not_require_rank_balance():
    labels = [18, 17, 0, 1, 11, 6, 16, 6, 3, 2, 6, 12, 2, 15, 6, 6]
    order = balance_accumulation(list(range(len(labels))), labels, 4)
    before = accumulation_metrics(labels, 4)
    after = accumulation_metrics([labels[i] for i in order], 4)
    assert after["std_global_update_labels"] < before["std_global_update_labels"]
    rank_before = np.asarray(labels).reshape(-1, 2, 2).sum(axis=1)
    rank_after = np.asarray([labels[i] for i in order]).reshape(-1, 2, 2).sum(axis=1)
    assert rank_after.std() > rank_before.std()


@pytest.mark.parametrize(
    "dp_replicate, dp_shard, expected",
    [(None, None, 32), (4, 1, 16), (1, 4, 16), (2, 2, 16), (1, 1, 4)],
)
def test_trainer_normalizes_data_parallel_update_size(dp_replicate, dp_shard, expected):
    from types import SimpleNamespace

    from axolotl.core.trainers.base import AxolotlTrainer

    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(world_size=8, gradient_accumulation_steps=4)
    parallelism = (
        SimpleNamespace(dp_replicate_size=dp_replicate, dp_shard_size=dp_shard)
        if dp_replicate is not None
        else None
    )
    trainer.accelerator = SimpleNamespace(parallelism_config=parallelism)
    assert trainer._batches_per_optimizer_step() == expected


def test_balance_labels_normalized_group_preserves_incomplete_update():
    from axolotl.utils.samplers.label_balance import balance_labels

    batches = [[[i]] for i in range(10)]
    result = balance_labels(
        batches,
        np.full(10, 4),
        np.array([1, 3] * 5),
        np.zeros(10, dtype=int),
        capacity=4,
        bin_size=1,
        seed=42,
        batches_per_optimizer_step=4,
    )
    assert result[8:] == batches[8:]
    assert sorted(i for batch in result for row in batch for i in row) == list(
        range(10)
    )
