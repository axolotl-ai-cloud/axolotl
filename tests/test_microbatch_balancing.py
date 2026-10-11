# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Within-update cost refinement preserves samples, masks, and rank schedules."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from axolotl.utils.samplers.microbatch_balance import balance_microbatches


def cost(batch, lengths, multiple):
    maximum = max(sum(int(lengths[i]) for i in row) for row in batch)
    return len(batch) * ((maximum + multiple - 1) // multiple * multiple)


def labels(batch, counts, starts):
    return sum(sum(int(counts[i]) for i in row) - int(starts[row[0]]) for row in batch)


def members(batches):
    return Counter(i for batch in batches for row in batch for i in row)


def masked_starts(batches, starts):
    return Counter(row[0] for batch in batches for row in batch if starts[row[0]])


def assert_preserved(before, after, lengths, counts, starts, width, multiple):
    full = len(before) // width * width
    assert after[full:] == before[full:]
    if full:
        assert (
            np.var([cost(b, lengths, multiple) for b in after[:full]])
            <= np.var([cost(b, lengths, multiple) for b in before[:full]]) + 1e-8
        )
    for offset in range(0, full, width):
        a, b = before[offset : offset + width], after[offset : offset + width]
        assert members(a) == members(b)
        assert masked_starts(a, starts) == masked_starts(b, starts)
        assert [[len(row) for row in batch] for batch in a] == [
            [len(row) for row in batch] for batch in b
        ]
        ac, bc = (
            [cost(x, lengths, multiple) for x in a],
            [cost(x, lengths, multiple) for x in b],
        )
        al, bl = (
            [labels(x, counts, starts) for x in a],
            [labels(x, counts, starts) for x in b],
        )
        assert sum(al) == sum(bl)
        assert np.var(bl) <= np.var(al) + 1e-8
        assert sum(bc) <= sum(ac)
        assert max(bc) <= max(ac)
        assert np.var(bc) <= np.var(ac) + 1e-8


@pytest.mark.parametrize("mode", ["packed", "flattened", "padded"])
@pytest.mark.parametrize("width", [1, 2, 4, 16])
@pytest.mark.parametrize("multiple", [1, 8])
def test_refinement_invariants(mode, width, multiple):
    rng = np.random.default_rng(42)
    size = (width * 3 + 1) * 4
    lengths = rng.integers(2, 33, size)
    counts = rng.integers(1, 3, size)
    starts = rng.integers(0, 2, size) if mode == "packed" else np.zeros(size, dtype=int)
    rows = np.arange(size).reshape(-1, 4).tolist()
    batches = (
        [[[i] for i in row] for row in rows]
        if mode == "padded"
        else [[row[:2], row[2:]] for row in rows]
        if mode == "packed"
        else [[row] for row in rows]
    )
    original = deepcopy(batches)
    after = balance_microbatches(
        batches, lengths, counts, starts, width, multiple, capacity=128
    )
    assert batches == original
    assert_preserved(batches, after, lengths, counts, starts, width, multiple)
    assert after == balance_microbatches(
        batches, lengths, counts, starts, width, multiple, capacity=128
    )
    if width == 1:
        assert after == batches


def test_strict_peak_improvement_without_update_membership_changes():
    lengths = np.array([9, 7, 3, 1, 8, 8, 2, 2, 5])
    batches = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8]]]
    counts, starts = np.ones(9, dtype=int), np.zeros(9, dtype=int)
    after = balance_microbatches(batches, lengths, counts, starts, 2, capacity=16)
    assert_preserved(batches, after, lengths, counts, starts, 2, 1)
    assert [cost(b, lengths, 1) for b in after] == [10, 10, 10, 10, 5]


def test_padded_refinement_rejects_added_padding():
    batches = [[[0], [1]], [[2], [3]]]
    lengths = np.array([10, 10, 1, 1])
    assert balance_microbatches(batches, lengths, np.ones(4), np.zeros(4), 2) == batches


def test_padded_refinement_balances_tokens_without_adding_padding():
    batches = [[[0], [1], [2]], [[3], [4], [5]]]
    lengths = np.array([10, 2, 2, 9, 6, 5])
    after = balance_microbatches(batches, lengths, np.ones(6), np.zeros(6), 2)
    assert [cost(b, lengths, 1) for b in after] == [30, 27]
    assert [sum(lengths[i] for row in b for i in row) for b in after] == [17, 17]


def test_packed_boundary_supervision_cannot_move_to_interior():
    lengths = np.array([9, 7, 3, 1])
    counts = np.ones(4, dtype=int)
    starts = np.array([1, 0, 0, 1])
    batches = [[[0, 1]], [[2, 3]]]
    after = balance_microbatches(batches, lengths, counts, starts, 2, capacity=16)
    assert_preserved(batches, after, lengths, counts, starts, 2, 1)


@pytest.mark.parametrize("width,multiple", [(0, 1), (2, 0)])
def test_invalid_refinement_settings(width, multiple):
    with pytest.raises(ValueError, match="positive"):
        balance_microbatches([], [], [], [], width, multiple)


@pytest.mark.parametrize("mode", ["packed", "flattened", "padded"])
@pytest.mark.parametrize("replicas,steps", [(1, 4), (2, 4), (4, 1)])
def test_sampler_refinement_preserves_actual_distributed_update_membership(
    monkeypatch, mode, replicas, steps
):
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler

    from axolotl.utils.samplers import (
        LabelBalancedRandomSampler,
        MultipackBatchSampler,
        label_balanced,
        multipack,
    )

    rng = np.random.default_rng(8)
    lengths = rng.integers(2, 33, 259)
    counts = rng.integers(0, 3, 259)
    starts = np.zeros(259, dtype=int)
    calls = []

    def record(batches, *args, **kwargs):
        before = deepcopy(batches)
        result = balance_microbatches(batches, *args, **kwargs)
        calls.append((before, result))
        return result

    monkeypatch.setattr(
        multipack if mode == "packed" else label_balanced,
        "balance_microbatches",
        record,
    )
    width = replicas * steps
    if mode == "packed":
        sampler = MultipackBatchSampler(
            list(range(259)),
            lengths=lengths,
            label_counts=counts,
            batch_size=1,
            batch_max_len=64,
            bin_size=32,
            num_processes=1,
            seed=42,
            batches_per_optimizer_step=width,
        )
        final = sampler.generate_batches()
        sampler._len_across_ranks = len(final)
        loader = sampler
    else:
        sampler = LabelBalancedRandomSampler(
            lengths,
            counts,
            4,
            seed=42,
            length_mode=mode,
            batches_per_optimizer_step=width,
        )
        list(sampler)
        loader = BatchSampler(sampler, 4, True)
    assert len(calls) == 1
    before, after = calls[0]
    assert_preserved(before, after, lengths, counts, starts, width, 1)
    shards = [
        list(
            BatchSamplerShard(
                loader, num_processes=replicas, process_index=r, even_batches=False
            )
        )
        for r in range(replicas)
    ]
    for update in range(len(before) // width):
        observed = []
        for shard in shards:
            for batch in shard[update * steps : (update + 1) * steps]:
                observed.extend(
                    i for row in batch for i in row
                ) if mode == "packed" else observed.extend(batch)
        assert Counter(observed) == members(
            before[update * width : (update + 1) * width]
        )
    a, b = sampler.label_metrics["before_microbatch"], sampler.label_metrics["after"]
    for key in [
        "mean_global_update_labels",
        "std_global_update_labels",
        "total_label_count",
    ]:
        assert a[key] == b[key]
    assert b["std_label_count"] <= a["std_label_count"] + 1e-8


@pytest.mark.parametrize("mode", ["padded", "flattened"])
def test_single_sample_microbatches_skip_refinement(monkeypatch, mode):
    from axolotl.utils.samplers import LabelBalancedRandomSampler, label_balanced

    def unexpected(*args, **kwargs):
        pytest.fail("No sample redistribution is possible at batch size one")

    monkeypatch.setattr(label_balanced, "balance_microbatches", unexpected)
    sampler = LabelBalancedRandomSampler(
        [2] * 17, [1] * 17, 1, length_mode=mode, batches_per_optimizer_step=4
    )
    assert sorted(sampler) == list(range(17))
