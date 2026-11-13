"""Dispatch rollout micro-batches when the trainer asks for them, not one ahead.

accelerate's ``DataLoaderDispatcher`` assembles micro-batch ``i + 1`` before it
yields micro-batch ``i``, to flag the last one. Rollout micro-batches are packed
from the rollout queue, which drops samples older than ``max_staleness`` policy
versions against the trainer's version *at the time of the pull*. Pulled a step
early, the first micro-batch of every optimizer step was checked against the
previous version, so it trained one version staler than ``max_staleness`` allows
(every micro-batch, with ``gradient_accumulation_steps: 1``). The rollout stream
ends with its own stop micro-batch (``progress.STOP_KEY``), so nothing needs the
lookahead.

The token-budget batchers pull one sample past a full micro-batch and place it in
the next one, which may belong to the next optimizer step; ``FreshCarry`` checks
that sample again when it is placed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import accelerate.data_loader as accelerate_data_loader
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.utils import (
    broadcast,
    find_batch_size,
    initialize_tensors,
    send_to_device,
)
from tree_recurrence.integrations.trl import ChainRow, ChainTokenBudgetBatcher

from axolotl.core.trainers.async_grpo.trl_compat import TokenBudgetBatcher, TreeRow


class OnDemandDispatcher(DataLoaderDispatcher):
    """``DataLoaderDispatcher`` that fetches each batch when it is consumed."""

    def __iter__(self) -> Iterator:
        if not self.split_batches:
            # each rank's share comes from separate fetches; keep accelerate's loop
            yield from super().__iter__()
            return
        self.begin()
        self.set_epoch(self.iteration)
        # every rank builds the iterator: DataLoader shares a seed across ranks
        main_iterator = self.base_dataloader.__iter__()
        self._stop_iteration = False
        batch_index = 0
        while True:
            batch, batch_info = self._fetch_batches(main_iterator)
            if batch_info[0] is None:
                break
            if self.state.process_index != 0:
                batch = initialize_tensors(batch_info[0])
            batch = send_to_device(
                batch, self.state.device, non_blocking=self._non_blocking
            )
            batch = broadcast(batch, from_process=0)
            batch_size = find_batch_size(batch) // self.state.num_processes
            batch = self.slice_fn(
                batch,
                slice(
                    self.state.process_index * batch_size,
                    (self.state.process_index + 1) * batch_size,
                ),
                process_index=self.state.process_index,
                num_processes=self.state.num_processes,
            )
            if batch_index >= self.skip_batches:
                yield batch
            batch_index += 1
        self.end_of_dataloader = True
        self._update_state_dict()
        self.iteration += 1
        self.end()


@contextlib.contextmanager
def on_demand_dispatch():
    """Make ``Accelerator.prepare`` build an ``OnDemandDispatcher`` for dispatched loaders."""
    original = accelerate_data_loader.DataLoaderDispatcher
    accelerate_data_loader.DataLoaderDispatcher = OnDemandDispatcher
    try:
        yield
    finally:
        accelerate_data_loader.DataLoaderDispatcher = original


class FreshCarry:
    """TRL's token-budget loop, re-checking the sample carried into a new micro-batch.

    Oversize samples never reach it: ``SampleFilterQueue`` drops them first.
    """

    row_cls: type

    def __iter__(self):
        queue = getattr(self.dataset, "queue", None)
        still_fresh = getattr(queue, "still_fresh", None)
        rows = [self.row_cls() for _ in range(self.num_processes)]
        for sample in self.dataset:
            while True:
                placed_row = self._place(rows, sample)
                if placed_row is not None:
                    placed_row.add(sample)
                    break
                yield rows
                rows = [self.row_cls() for _ in range(self.num_processes)]
                if still_fresh is not None and not still_fresh(sample["group_id"]):
                    queue.drop_held(sample["group_id"])
                    break


class FreshTokenBudgetBatcher(FreshCarry, TokenBudgetBatcher):
    row_cls = TreeRow


class FreshChainTokenBudgetBatcher(FreshCarry, ChainTokenBudgetBatcher):
    row_cls = ChainRow
