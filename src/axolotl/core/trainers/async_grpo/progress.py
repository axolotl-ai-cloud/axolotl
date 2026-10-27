"""Progress accounting and termination for async GRPO's rollout stream.

TRL's rollout worker cycles the prompts forever, and TRL ends an epoch-bounded run
only when an optimizer step ends. Rollouts dropped before training (no reward signal,
truncated, too stale, too long) complete no step, so a run whose remaining rollouts
are all dropped would generate forever.

``SampleFilterQueue`` drops the rollouts that would train nothing. Every drop and
every trained micro-batch reports to ``RolloutProgress``, which keeps
TRL's set of consumed prompt groups (read for the epoch bound and for resume). When
the run's prompt groups are consumed, or a full pass over the prompts trains nothing,
the rank-0 pipeline raises ``RolloutStop``. ``StopAwareDataset`` turns it into one
stop micro-batch that accelerate's dispatcher hands to every rank, so all ranks end
training (or raise the stall error) at the same micro-batch.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from axolotl.core.trainers.async_grpo.rollout import GROUP_ROWS
from axolotl.core.trainers.async_grpo.trl_compat import DataCollatorForRollout
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

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


class SampleFilterQueue:
    """A view of the rollout queue that drops samples which would train nothing.

    - ``stale``: more than ``max_staleness`` policy versions old, TRL's rule, applied
      here before TRL's dataset sees the sample so that the drop is accounted;
    - ``oversize``: longer than ``token_budget``, the rule of TRL's batchers;
    - ``zero_adv``: a zero advantage adds nothing to the policy loss but costs a full
      forward and backward (``skip_zero_advantage_batches``);
    - ``truncated``: masked out of the loss under ``mask_truncated_completions``.

    A group whose every row was dropped counts as consumed in ``progress``, so
    epochs and resume stay keyed on the prompts generated, and may end the stream
    (see the module doc). Rows are dropped one by one, so a group is settled
    once all of them arrived; one that keeps a row is counted when it trains.
    """

    def __init__(
        self,
        inner,
        progress: RolloutProgress,
        metrics: dict[str, list],
        *,
        model_version: Callable[[], int] | None = None,
        max_staleness: int | None = None,
        token_budget: Callable[[], int] = lambda: 0,
        skip_zero_advantage: bool = False,
        stop_token_ids: frozenset[int] | None = None,
    ):
        self._inner = inner
        self._progress = progress
        self._metrics = metrics
        self._model_version = model_version
        self._max_staleness = max_staleness
        self._token_budget = token_budget
        self._skip_zero_advantage = skip_zero_advantage
        self._stop_token_ids = stop_token_ids
        # group_id -> (rows seen, whether one was kept), for groups still arriving
        self._arriving: dict[int, tuple[int, bool]] = {}

    def qsize(self) -> int:
        return self._inner.qsize()

    def _drop_reason(self, sample) -> str | None:
        if (
            self._model_version is not None
            and self._max_staleness is not None
            and self._model_version() - sample.model_version > self._max_staleness
        ):
            return "stale"
        budget = self._token_budget()
        if budget > 0 and len(sample.input_ids) > budget:
            LOG.warning_once(
                f"Dropping rollout samples longer than token_budget={budget}; raise "
                "token_budget to keep them (counted in sample/dropped_oversize_total)."
            )
            return "oversize"
        if self._stop_token_ids is not None and (
            len(sample.input_ids) == 0
            or int(sample.input_ids[-1]) not in self._stop_token_ids
        ):
            return "truncated"
        if self._skip_zero_advantage and sample.advantage == 0.0:
            return "zero_adv"
        return None

    def get(self, timeout: float | None = None):
        self._progress.check()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            sample = self._inner.get(timeout=remaining)
            group_rows = sample.metrics.pop(GROUP_ROWS, None)
            reason = self._drop_reason(sample)
            self._metrics["sample/dropped_frac"].append(
                (float(reason is not None), 1.0)
            )
            if reason is not None:
                self._metrics[f"sample/dropped_{reason}_total"].append(1.0)
            self._settle(sample.group_id, group_rows, reason)
            if reason is None:
                return sample

    def _settle(self, group_id: int, group_rows: int | None, reason: str | None):
        """Report ``group_id`` as dropped once its last row arrived and none was kept.

        Rows without a group size (loops that do not record it) settle alone.
        """
        if group_rows is None:
            if reason is not None:
                self._progress.drop(group_id, reason)
            return
        seen, kept = self._arriving.pop(group_id, (0, False))
        seen, kept = seen + 1, kept or reason is None
        if seen < group_rows:
            self._arriving[group_id] = (seen, kept)
        elif not kept and reason is not None:
            self._progress.drop(group_id, reason)
