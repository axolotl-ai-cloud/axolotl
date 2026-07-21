"""Unit tests for MultipackBatchSampler drop_last total_slots accounting."""

from __future__ import annotations

import numpy as np
from torch.utils.data import SequentialSampler

from axolotl.utils.samplers.multipack import MultipackBatchSampler


class _ListDataset:
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


def _make_sampler(
    lengths: list[int],
    *,
    batch_size: int,
    batch_max_len: int,
    drop_last: bool,
    sequential: bool = False,
) -> MultipackBatchSampler:
    dataset = _ListDataset(len(lengths))
    return MultipackBatchSampler(
        sampler=SequentialSampler(dataset),
        lengths=np.array(lengths, dtype=np.int32),
        batch_size=batch_size,
        batch_max_len=batch_max_len,
        bin_size=max(1, len(lengths)),
        group_size=max(1, len(lengths)),
        sequential=sequential,
        drop_last=drop_last,
        num_count_samples=1,
        num_processes=1,
        safe_mode=True,
    )


def test_drop_last_reduces_total_token_slots_by_dropped_bins():
    # lengths=4, capacity=8 => two sequences per bin via first-fit.
    # 5 sequences => bins [[0,1],[2,3],[4]] (3 bins).
    # batch_size=2 => batches [full, incomplete]; incomplete last batch dropped.
    lengths = [4] * 5
    batch_size = 2
    batch_max_len = 8

    sampler = _make_sampler(
        lengths,
        batch_size=batch_size,
        batch_max_len=batch_max_len,
        drop_last=True,
        sequential=False,
    )
    batches = sampler.generate_batches(set_stats=True)

    assert len(batches) == 1
    assert len(batches[0]) == batch_size

    # Kept bins: 2; dropped bins: 1. total_token_slots must shrink by dropped bins.
    assert sampler.total_token_slots == 2 * batch_max_len


def test_drop_last_old_stale_reference_would_not_adjust():
    # Without the fix, total_slots adjustment used batches[-1] AFTER the drop,
    # which pointed at a full batch and subtracted 0. Assert we actually subtract.
    lengths = [4] * 5
    batch_max_len = 8
    sampler = _make_sampler(
        lengths,
        batch_size=2,
        batch_max_len=batch_max_len,
        drop_last=True,
        sequential=False,
    )
    sampler.generate_batches(set_stats=True)
    # Pre-drop slots would be 3 * batch_max_len; post-fix must be 2 * batch_max_len.
    assert sampler.total_token_slots == 2 * batch_max_len
    assert sampler.total_token_slots != 3 * batch_max_len


def test_drop_last_single_incomplete_batch_returns_empty_without_indexerror():
    lengths = [4]
    sampler = _make_sampler(
        lengths,
        batch_size=2,
        batch_max_len=8,
        drop_last=True,
        sequential=False,
    )
    batches = sampler.generate_batches(set_stats=True)
    assert batches == []
    assert sampler.total_token_slots == 0
