"""Publish the newest LoRA snapshot while training continues; skip what falls behind.

Two staging slots: one being published, one holding the newest waiting
snapshot. A snapshot that arrives while another waits replaces it (latest
wins) and the replaced step is counted as coalesced. Nothing queues up.

Ranks must publish the same step. Before each publish the publisher threads
agree on a target, the newest step any rank holds; a rank that has not staged
it yet keeps its slot for it until it arrives. Training reaches every snapshot
step on every rank, so the ranks converge; if one rank no longer holds the
target, everyone gives it up and agrees again.

A recovery save can hold a slot while it writes the same snapshot to disk. A
held slot is never overwritten; once the publisher is done with it, it stays
``HELD`` until the last hold is released.
"""

import threading
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

FREE, WRITING, WAITING, BUILDING, HELD = (
    "free",
    "writing",
    "waiting",
    "building",
    "held",
)


class SlotTable:
    """Slot states; the caller holds the lock."""

    def __init__(self, count: int = 2):
        self.states = [FREE] * count
        self.steps: list[int | None] = [None] * count
        self.holds = [0] * count
        self.target: int | None = None
        self.coalesced = 0

    def acquire(self, step: int) -> int | None:
        """A slot to stage ``step`` into, replacing an older waiting snapshot first."""
        replaceable = [
            i
            for i, state in enumerate(self.states)
            if state == WAITING and self.steps[i] != self.target and not self.holds[i]
        ]
        if replaceable:
            slot = min(replaceable, key=lambda i: self.steps[i] or 0)
            self.coalesced += 1
        elif FREE in self.states:
            slot = self.states.index(FREE)
        else:
            self.coalesced += 1
            return None
        self.states[slot], self.steps[slot] = WRITING, step
        return slot

    def written(self, slot: int):
        self.states[slot] = WAITING

    def newest(self) -> int | None:
        waiting = [
            s
            for s, state in zip(self.steps, self.states, strict=True)
            if state == WAITING and s is not None
        ]
        return max(waiting) if waiting else None

    def has(self, step: int) -> bool:
        return any(
            state == WAITING and s == step
            for s, state in zip(self.steps, self.states, strict=True)
        )

    def take(self, step: int) -> int:
        """Mark ``step`` building and drop older waiting snapshots."""
        slot = next(
            i
            for i, state in enumerate(self.states)
            if state == WAITING and self.steps[i] == step
        )
        self.states[slot] = BUILDING
        for i, (state, waiting) in enumerate(zip(self.states, self.steps, strict=True)):
            if state == WAITING and waiting is not None and waiting < step:
                self.release(i)
                self.coalesced += 1
        self.target = None
        return slot

    def release(self, slot: int):
        if self.holds[slot]:
            self.states[slot] = HELD
        else:
            self.states[slot], self.steps[slot] = FREE, None

    def hold(self, slot: int):
        self.holds[slot] += 1

    def unhold(self, slot: int):
        self.holds[slot] -= 1
        if not self.holds[slot] and self.states[slot] == HELD:
            self.states[slot], self.steps[slot] = FREE, None

    def drop_waiting(self) -> int:
        dropped = 0
        for i, state in enumerate(self.states):
            if state == WAITING:
                self.release(i)
                dropped += 1
        self.coalesced += dropped
        return dropped


class CoalescingScheduler:
    """Run ``publish(buffers, step)`` on a background thread for the newest snapshot.

    ``offer`` and ``wait_training`` run on the training thread; ``publish``,
    ``restore`` and every ``comm`` collective run on the publisher thread only.
    """

    def __init__(
        self,
        stager,
        publish: Callable[[Sequence[torch.Tensor], int], dict] | None,
        comm,
        *,
        restore: Callable[[Sequence[torch.Tensor], int], Any] | None = None,
        arrival_timeout: float = 600.0,
    ):
        self.stager, self.publish_fn, self.comm = stager, publish, comm
        self.restore_fn = restore
        self.arrival_timeout = arrival_timeout
        self.table = SlotTable(len(stager.buffers))
        self.cond = threading.Condition()
        self.metrics: deque[dict] = deque()
        self.error: BaseException | None = None
        self._closing = False
        self._flush = True
        self._reported = 0
        self._restore: tuple[int, int] | None = None
        self.thread: threading.Thread | None = None

    def start(self, restore_step: int | None = None):
        """Start the publisher thread; with ``restore_step``, stage and restore first."""
        if self.publish_fn is None:
            raise ValueError("A scheduler without a publish function only stages")
        if restore_step is not None and self.restore_fn is not None:
            with self.cond:
                slot = self.table.acquire(restore_step)
            assert slot is not None
            self.stager.stage(slot)
            with self.cond:
                self.table.states[slot] = BUILDING
            self._restore = (slot, restore_step)
        self.thread = threading.Thread(
            target=self._run, name="checkpoint-publish", daemon=True
        )
        self.thread.start()

    @property
    def coalesced(self) -> int:
        return self.table.coalesced

    def offer(self, step: int) -> bool:
        """Stage the current LoRA as ``step``; ``False`` if it had to be skipped."""
        return self.snapshot(step) is not None

    def snapshot(
        self, step: int, *, publish: bool = True, hold: bool = False
    ) -> int | None:
        """Stage the current LoRA as ``step`` and return its slot, or ``None`` if skipped.

        With ``publish`` the snapshot is offered to the publisher; with ``hold`` the
        slot stays readable until ``unhold``. A slot neither published nor held
        would be freed at once, so one of the two is required.
        """
        if not (publish or hold):
            raise ValueError("A snapshot must be published or held")
        self.check()
        with self.cond:
            if publish and self._closing:
                return None
            slot = self.table.acquire(step)
            if slot is None:
                return None
            if hold:
                self.table.hold(slot)
        try:
            self.stager.stage(slot)
        except BaseException:
            with self.cond:
                if hold:
                    self.table.unhold(slot)
                self.table.release(slot)
            raise
        with self.cond:
            if publish:
                self.table.written(slot)
            else:
                self.table.release(slot)
            self.cond.notify_all()
        return slot

    def buffers(self, slot: int) -> Sequence[torch.Tensor]:
        """A held slot's staged buffers, once the copy has finished."""
        return self.stager.ready(slot)

    def unhold(self, slot: int):
        with self.cond:
            self.table.unhold(slot)
            self.cond.notify_all()

    def wait_training(self):
        self.stager.wait_training()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Checkpoint publishing failed") from self.error

    def drain_metrics(self) -> list[dict]:
        out = []
        while self.metrics:
            out.append(self.metrics.popleft())
        return out

    def close(self, flush: bool = True, timeout: float | None = None):
        with self.cond:
            self._closing, self._flush = True, flush
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join(timeout)
        self.check()

    def _run(self):
        try:
            if self._restore is not None:
                slot, step = self._restore
                self.restore_fn(self.stager.ready(slot), step)
                with self.cond:
                    self.table.release(slot)
            while self._round():
                pass
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            LOG.exception("Checkpoint publisher thread failed")
            self.error = exc

    def _round(self) -> bool:
        with self.cond:
            self.cond.wait_for(lambda: self.table.newest() is not None or self._closing)
            if self._closing and not self._flush:
                self.table.drop_waiting()
            newest, closing = self.table.newest(), self._closing
        proposals = self.comm.all_gather_object((newest, closing))
        steps = [step for step, _ in proposals if step is not None]
        all_closing = all(flag for _, flag in proposals)
        if not steps:
            return not all_closing
        target = max(steps)
        with self.cond:
            self.table.target = target
            self.cond.wait_for(
                lambda: (
                    self.table.has(target)
                    or (self.table.newest() or -1) > target
                    or (self._closing and newest == self.table.newest())
                ),
                timeout=self.arrival_timeout,
            )
            have = self.table.has(target)
        haves = self.comm.all_gather_object(have)
        if not all(haves):
            with self.cond:
                self.table.target = None
                if all_closing:
                    self.table.drop_waiting()
            LOG.warning("Ranks did not all stage step %d; skipping it", target)
            return True
        with self.cond:
            slot = self.table.take(target)
        assert self.publish_fn is not None
        try:
            metrics = self.publish_fn(self.stager.ready(slot), target)
        finally:
            with self.cond:
                self.table.release(slot)
        coalesced = self.table.coalesced
        metrics = dict(metrics or {})
        metrics.update(step=target, coalesced_steps=coalesced - self._reported)
        self._reported = coalesced
        self.metrics.append(metrics)
        return True
