"""Async GRPO's train loader dispatches each micro-batch when it is consumed."""

from collections import defaultdict

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from async_test_utils import _sample
from torch.utils.data import DataLoader, IterableDataset

from axolotl.core.trainers.async_grpo.dispatch import (
    FreshTokenBudgetBatcher,
    OnDemandDispatcher,
    on_demand_dispatch,
)


class Counting(IterableDataset):
    def __init__(self, length):
        self.length, self.pulled = length, 0

    def __iter__(self):
        for i in range(self.length):
            self.pulled += 1
            yield torch.tensor([i])


def prepare(dataset, on_demand):
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=True, split_batches=True
        ),
    )
    loader = DataLoader(dataset, batch_size=1)
    if on_demand:
        with on_demand_dispatch():
            return accelerator.prepare(loader)
    return accelerator.prepare(loader)


def test_stock_dispatcher_pulls_a_batch_ahead():
    dataset = Counting(4)
    iterator = iter(prepare(dataset, on_demand=False))
    next(iterator)
    assert dataset.pulled == 2


def test_on_demand_dispatcher_pulls_only_what_was_consumed():
    dataset = Counting(3)
    loader = prepare(dataset, on_demand=True)
    assert type(loader) is OnDemandDispatcher
    iterator = iter(loader)
    values = []
    for expected in range(3):
        values.append(int(next(iterator)))
        assert dataset.pulled == expected + 1
    assert list(iterator) == [] and values == [0, 1, 2]
    assert loader.end_of_dataloader
    with on_demand_dispatch():
        pass
    assert type(prepare(Counting(1), on_demand=False)) is not OnDemandDispatcher


class HeldQueue:
    """The ``still_fresh`` / ``drop_held`` side of ``SampleFilterQueue``."""

    def __init__(self):
        self.version, self.versions, self.dropped = 0, {}, []

    def still_fresh(self, group_id):
        return self.version - self.versions[group_id] <= 0

    def drop_held(self, group_id, rows=1):
        self.dropped.append((group_id, rows))


class Source:
    def __init__(self, samples, queue):
        self.samples, self.queue = samples, queue

    def __iter__(self):
        for sample in self.samples:
            self.queue.versions[sample["group_id"]] = self.queue.version
            yield sample


def test_token_budget_batcher_rechecks_the_sample_it_carries():
    queue = HeldQueue()
    # one sample per micro-batch: each pulls the next before it can yield
    samples = [
        _sample([10 * group_id + i for i in range(1, 9)], 4, group_id)
        for group_id in range(4)
    ]
    batcher = FreshTokenBudgetBatcher(Source(samples, queue), 1, 8, defaultdict(list))
    iterator = iter(batcher)
    first = next(iterator)
    assert [s["group_id"] for s in first[0].samples] == [0]
    # an optimizer step happens before the carried group 1 is placed
    queue.version = 1
    second = next(iterator)
    assert queue.dropped == [(1, 1)]
    assert [s["group_id"] for s in second[0].samples] == [2]
