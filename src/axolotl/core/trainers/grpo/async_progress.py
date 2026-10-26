"""Progress accounting and termination for async GRPO's rollout stream.

TRL's rollout worker cycles the prompts forever, and TRL ends an epoch-bounded run
only when an optimizer step ends. Rollouts dropped before training (no reward signal,
truncated, too stale, too long) complete no step, so a run whose remaining rollouts
are all dropped would generate forever.

Every drop and every trained micro-batch reports to ``RolloutProgress``, which keeps
TRL's set of consumed prompt groups (read for the epoch bound and for resume). When
the run's prompt groups are consumed, or a full pass over the prompts trains nothing,
the rank-0 pipeline raises ``RolloutStop``. ``StopAwareDataset`` turns it into one
stop micro-batch that accelerate's dispatcher hands to every rank, so all ranks end
training (or raise the stall error) at the same micro-batch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout

STOP_KEY = "rollout_stop"
STOP_DONE = 1
STOP_STALLED = 2

_REASONS = {
    "zero_adv": (
        "had a zero advantage",
        "make the rewards vary within a group, or set skip_zero_advantage_batches: "
        "false",
    ),
    "truncated": (
        "were truncated",
        "raise max_completion_length, or unset mask_truncated_completions",
    ),
    "stale": ("were too stale", "raise max_staleness, or lower max_inflight_tasks"),
    "oversize": (
        "did not fit token_budget",
        "raise token_budget (and gradient_accumulation_steps when rewards are "
        "scored on the trainer)",
    ),
}


class RolloutStop(Exception):
    """Ends the rollout stream; ``error`` is set when the run made no progress."""

    def __init__(self, error: str | None = None):
        super().__init__(error or "the run's prompt groups are consumed")
        self.error = error


class RolloutProgress:
    """Consumed and trained prompt groups, and the stream's stop conditions.

    ``consumed`` is the trainer's set TRL counts epochs and resumes from: a group
    counts once it trained or its rollouts were dropped. ``stall_groups`` bounds how
    many groups may be consumed in a row without one training.

    Only rank 0 reports, from the rollout stream and, with trainer-side scoring,
    from ``score_step``; the stream raises the stop, so that it reaches every rank
    as a stop micro-batch.
    """

    def __init__(
        self,
        consumed: set[int],
        *,
        epochs_done: Callable[[], bool],
        stall_groups: int | None,
    ):
        self.consumed = consumed
        self.trained: set[int] = set()
        self.dropped: Counter[str] = Counter()
        self.stop: RolloutStop | None = None
        self._epochs_done = epochs_done
        self._stall_groups = stall_groups
        self._since_trained: Counter[str] = Counter()

    def drop(self, group_id: int, reason: str) -> None:
        """Count ``group_id`` as consumed without training; may end the stream."""
        if group_id in self.consumed:
            return
        self.consumed.add(group_id)
        self._untrained(reason)
        self.check()
        if self._epochs_done():
            self._raise(RolloutStop())

    def exclude(self, group_id: int, reason: str) -> None:
        """Count a collated group whose every sample scoring took out of the loss.

        Runs outside the stream, so the stall it may complete is raised by the
        stream's next ``check``.
        """
        self.consumed.add(group_id)
        self._untrained(reason)

    def check(self) -> None:
        """End the stream when a full pass over the prompts trained no group."""
        if self._stall_groups and self._since_trained.total() >= self._stall_groups:
            self._raise(RolloutStop(self._stall_message()))

    def train(self, group_ids: Iterable[int]) -> None:
        """Record groups that trained."""
        group_ids = set(group_ids)
        if group_ids:
            self.consumed.update(group_ids)
            self.trained.update(group_ids)
            self._since_trained.clear()

    def _untrained(self, reason: str) -> None:
        self.dropped[reason] += 1
        self._since_trained[reason] += 1

    def _raise(self, stop: RolloutStop) -> None:
        self.stop = stop
        raise stop

    def _stall_message(self) -> str:
        reasons = self._since_trained.most_common()
        counts = ", ".join(
            f"{count} {_REASONS[reason][0]}" for reason, count in reasons
        )
        hints = "; ".join(_REASONS[reason][1] for reason, _ in reasons)
        return (
            f"No rollout group trained in the last {self._since_trained.total()} "
            f"groups, a full pass over the prompts: {counts}. To fix: {hints}."
        )


class StopAwareDataset(torch.utils.data.IterableDataset):
    """A rollout batcher whose stream ends with the ``RolloutStop`` it raised."""

    def __init__(self, batcher):
        self.batcher = batcher

    def __iter__(self) -> Iterator[Any]:
        try:
            yield from self.batcher
        except RolloutStop as stop:
            yield stop


@dataclass
class ProgressCollator(DataCollatorForRollout):
    """TRL's collator, reporting trained groups and passing a stream stop on."""

    progress: RolloutProgress | None = None

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        (rows,) = examples
        if isinstance(rows, RolloutStop):
            code = STOP_DONE if rows.error is None else STOP_STALLED
            return {STOP_KEY: torch.full((self.num_processes, 1), code)}
        batch = super().torch_call(examples)
        if self.progress is not None:
            self.progress.train(s["group_id"] for row in rows for s in row.samples)
        return batch


__all__ = [
    "STOP_DONE",
    "STOP_KEY",
    "STOP_STALLED",
    "ProgressCollator",
    "RolloutProgress",
    "RolloutStop",
    "StopAwareDataset",
]
