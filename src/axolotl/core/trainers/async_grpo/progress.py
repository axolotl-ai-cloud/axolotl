"""Progress accounting and termination for async GRPO's rollout stream.

TRL's rollout worker cycles the prompts forever, and TRL ends an epoch-bounded run
only when an optimizer step ends. Rollouts dropped before training (no reward signal,
truncated, too stale, too long) complete no step, so a run whose remaining rollouts
are all dropped would generate forever.

``SampleFilterQueue`` drops the rollouts that would train nothing. Every drop and
every trained micro-batch reports to ``RolloutProgress``, which keeps TRL's sets of
trained and dropped prompt groups (their union is read for the epoch bound and for
resume). When the run's prompt groups are consumed, or a full pass over the prompts
trains nothing,
the rank-0 pipeline raises ``RolloutStop``. ``StopAwareDataset`` turns it into one
stop micro-batch that accelerate's dispatcher hands to every rank, so all ranks end
training (or raise the stall error) at the same micro-batch.
"""

from __future__ import annotations

import itertools
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

# Kept through a prune; far more groups than a carried row can trail by.
_RECENT_VERSIONS = 4096

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
    """Trained and dropped prompt groups, and the stream's stop conditions.

    ``trained`` and ``dropped`` are the trainer's ``_trained_groups`` and
    ``_dropped_groups``, whose union TRL counts epochs and resumes from: a group
    counts once it trained or its rollouts were dropped. TRL's collator adds every
    group it packs to ``trained``. ``drop_reasons`` counts the dropped groups by
    reason, and ``stall_groups`` bounds how many groups may be consumed in a row
    without one training.

    Only rank 0 reports, from the rollout stream and, with trainer-side scoring,
    from ``score_step``; the stream raises the stop, so that it reaches every rank
    as a stop micro-batch.
    """

    def __init__(
        self,
        trained: set[int],
        dropped: set[int],
        *,
        epochs_done: Callable[[], bool],
        stall_groups: int | None,
    ):
        self.trained = trained
        self.dropped = dropped
        self.drop_reasons: Counter[str] = Counter()
        self.stop: RolloutStop | None = None
        self._epochs_done = epochs_done
        self._stall_groups = stall_groups
        self._since_trained: Counter[str] = Counter()

    def drop(self, group_id: int, reason: str) -> None:
        """Count ``group_id`` as consumed without training; may end the stream."""
        if group_id in self.trained or group_id in self.dropped:
            return
        self.dropped.add(group_id)
        self._untrained(reason)
        self.check()
        if self._epochs_done():
            self._raise(RolloutStop())

    def exclude(self, group_id: int, reason: str) -> None:
        """Count a collated group whose every sample scoring took out of the loss.

        Runs outside the stream, so the stall it may complete is raised by the
        stream's next ``check``.
        """
        self.trained.discard(group_id)
        self.dropped.add(group_id)
        self._untrained(reason)

    def check(self) -> None:
        """End the stream when a full pass over the prompts trained no group."""
        if self._stall_groups and self._since_trained.total() >= self._stall_groups:
            self._raise(RolloutStop(self._stall_message()))

    def train(self, group_ids: Iterable[int]) -> None:
        """Record groups that trained."""
        group_ids = set(group_ids)
        if group_ids:
            self.trained.update(group_ids)
            self._since_trained.clear()

    def _untrained(self, reason: str) -> None:
        self.drop_reasons[reason] += 1
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
    once all of them arrived; one that keeps a row is counted when it trains. The
    rollout process sends a group whose every advantage is zero as its bare
    ``int`` id, which is settled as dropped right away.

    A batcher that holds a kept row over into a later micro-batch re-checks it
    with ``still_fresh`` there and reports a drop with ``drop_held``, since the
    policy may have moved on in between.
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
        # group_id -> (rows seen, rows kept), for groups still arriving
        self._arriving: dict[int, tuple[int, int]] = {}
        # group_id -> rows kept and not yet dropped, for arrived groups that kept one
        self._held: dict[int, int] = {}
        # group_id -> its policy version, for groups that kept a row
        self._versions: dict[int, int] = {}
        self._prune_at = 4096

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
            if isinstance(sample, int):
                self._metrics["sample/dropped_zero_adv_groups_total"].append(1.0)
                self._progress.drop(sample, "zero_adv")
                continue
            group_rows = sample.metrics.pop(GROUP_ROWS, None)
            reason = self._drop_reason(sample)
            self._metrics["sample/dropped_frac"].append(
                (float(reason is not None), 1.0)
            )
            if reason is not None:
                self._metrics[f"sample/dropped_{reason}_total"].append(1.0)
            self._settle(sample.group_id, group_rows, reason)
            if reason is None:
                # Newest last, so a prune keeps the versions carried rows still need.
                self._versions.pop(sample.group_id, None)
                self._versions[sample.group_id] = sample.model_version
                if len(self._versions) > self._prune_at:
                    self._prune()
                return sample

    def _settle(self, group_id: int, group_rows: int | None, reason: str | None):
        """Report ``group_id`` as dropped once its last row arrived and none was kept.

        Rows without a group size (loops that do not record it) settle alone.
        """
        if group_rows is None:
            if reason is not None:
                self._progress.drop(group_id, reason)
            return
        seen, kept = self._arriving.pop(group_id, (0, 0))
        seen, kept = seen + 1, kept + (reason is None)
        if seen < group_rows:
            self._arriving[group_id] = (seen, kept)
        elif kept:
            self._hold(group_id, kept)
        elif reason is not None:
            self._progress.drop(group_id, reason)

    def _hold(self, group_id: int, kept: int):
        if len(self._held) > self._prune_at:
            self._prune()
        self._held[group_id] = kept

    def _prune(self):
        """Forget settled groups; rows without a group size never reach ``_hold``.

        The newest versions stay: a settled group can still have a row carried into
        the next micro-batch, which ``still_fresh`` must judge.
        """
        settled = self._progress.trained | self._progress.dropped
        self._held = {g: n for g, n in self._held.items() if g not in settled}
        recent = set(itertools.islice(reversed(self._versions), _RECENT_VERSIONS))
        self._versions = {
            g: v
            for g, v in self._versions.items()
            if g in recent or g in self._held or g in self._arriving or g not in settled
        }
        self._prune_at = max(4096, 2 * max(len(self._held), len(self._versions)))

    def still_fresh(self, group_id: int) -> bool:
        """Whether a kept row of ``group_id`` is within ``max_staleness`` now."""
        version = self._versions.get(group_id)
        if version is None or self._model_version is None:
            return True
        if self._max_staleness is None:
            return True
        return self._model_version() - version <= self._max_staleness

    def drop_held(self, group_id: int, rows: int = 1):
        """Drop ``rows`` kept rows of ``group_id`` that went stale before placement."""
        self._metrics["sample/dropped_stale_total"].append(float(rows))
        if group_id in self._arriving:
            seen, kept = self._arriving[group_id]
            self._arriving[group_id] = (seen, max(kept - rows, 0))
            return
        if group_id not in self._held:
            # A version without a held count is a row without a group size, which
            # settles alone like its early drops in ``_settle``.
            if group_id in self._versions:
                self._progress.drop(group_id, "stale")
            return
        kept = self._held.pop(group_id) - rows
        if kept > 0:
            self._held[group_id] = kept
        else:
            self._progress.drop(group_id, "stale")
