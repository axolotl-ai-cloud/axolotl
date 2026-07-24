"""Tests for MultipackBatchSampler.generate_batches drop_last accounting."""

import numpy as np

from axolotl.utils.samplers import multipack
from axolotl.utils.samplers import MultipackBatchSampler


def _make_sampler(num_items, batch_size, batch_max_len, drop_last):
    return MultipackBatchSampler(
        sampler=list(range(num_items)),
        lengths=np.ones(num_items, dtype=np.int32),
        batch_size=batch_size,
        batch_max_len=batch_max_len,
        bin_size=batch_max_len,
        sequential=False,
        drop_last=drop_last,
    )


def test_drop_last_subtracts_dropped_bins_from_slots(monkeypatch):
    # 7 bins, batch_size 3 -> batches of [3, 3, 1]; the final batch of 1 bin is dropped.
    batch_max_len = 100
    monkeypatch.setattr(
        multipack, "pack_parallel", lambda lengths, **_: [[i] for i in range(7)]
    )

    sampler = _make_sampler(
        num_items=7, batch_size=3, batch_max_len=batch_max_len, drop_last=True
    )
    batches = sampler.generate_batches(set_stats=True)

    # Only the two full batches survive.
    assert [len(b) for b in batches] == [3, 3]
    # 6 remaining bins each contribute batch_max_len slots; the dropped bin is removed.
    assert sampler.total_token_slots == 6 * batch_max_len
    # The dropped bin's token is also removed from the used-token total.
    assert sampler.total_tokens_used == 6


def test_drop_last_does_not_crash_on_single_incomplete_batch(monkeypatch):
    # 1 bin, batch_size 3 -> a single incomplete batch that is dropped entirely.
    monkeypatch.setattr(multipack, "pack_parallel", lambda lengths, **_: [[0]])

    sampler = _make_sampler(
        num_items=1, batch_size=3, batch_max_len=100, drop_last=True
    )
    batches = sampler.generate_batches(set_stats=True)

    assert batches == []
    assert sampler.total_token_slots == 0
    # Dropping the only bin removes its token too, leaving nothing used.
    assert sampler.total_tokens_used == 0
