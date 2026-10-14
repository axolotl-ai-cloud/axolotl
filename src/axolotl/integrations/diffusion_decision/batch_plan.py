"""Per-epoch training batch plans over the cached decision canvas layer.

The prepared-row cache holds a *canvas layer* that no batch setting touches: per
source, every decontaminated candidate record, with the canvas of each record in
the epoch-0 capped subset already built. A :class:`DecisionBatchPlan` turns it into
training draws at startup (and again for every epoch):

- budget filter over the cached row lengths (``micro_batch_size`` decides the
  packed physical budget),
- per epoch: the capped subset (the next ``max_examples_per_source`` window of each
  source), a fresh mixture draw and stratified grouping at ``micro_batch_size``,
- per draw: the canvas. A record's first draw in epoch 0 is the cached canvas; every
  other draw rebuilds it lazily with question/choice order seeded by the record
  identity, the epoch and the draw number of that record within the epoch, by
  concatenating the record's pre-tokenized segments where it has them.

Every epoch has the first epoch's batch count, so the trainer's step math holds.
Everything is a pure function of (canvas layer, seed, batch settings, epoch), so
ranks and restarts agree on the batches of any epoch.
"""

from __future__ import annotations

import json
import logging
import random
import zlib
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._util import stable_seed
from .budgets import decision_budget, exceeds_budget
from .mixture import DeterministicMixtureSampler, cap_window

LOG = logging.getLogger(__name__)
_EPOCH_PLANS_KEPT = 3


@dataclass
class SourcePool:
    """One source of the canvas layer, aligned with its candidate records.

    ``entries[i]`` is the prepared row of candidate ``i`` (epoch-0 capped subset),
    None when its canvas overflowed, or ``bytes`` (zlib JSON of the record) for a
    candidate outside the epoch-0 subset, built only when a later epoch draws it.
    ``records[i]`` is the original (unpermuted) record of a prepared entry.
    ``segments[i]`` (when present) holds candidate ``i``'s pre-tokenized canvas
    segments (segments.py), so a re-permuted draw concatenates cached token spans
    instead of re-tokenizing; None where a record keeps full tokenization.
    """

    source: str
    cap: int | None
    entries: list[Any]
    records: list[Any]
    segments: list[Any] | None = None


def encode_reserve(record: Mapping[str, Any]) -> bytes:
    return zlib.compress(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        1,
    )


def decode_reserve(data: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(data).decode("utf-8"))


@dataclass(frozen=True)
class BudgetLimits:
    logical: int | None
    physical: int | None


def budget_reason(row: Mapping[str, Any], limits: BudgetLimits, layout: Any):
    canvas = row["canvas"]
    return exceeds_budget(
        decision_budget(
            prompt_tokens=len(canvas.prompt_ids),
            canvas_tokens=len(canvas.canvas_ids),
            layout=layout,
        ),
        logical_limit=limits.logical,
        physical_limit=limits.physical,
    )


@dataclass(frozen=True)
class _EpochDraws:
    source: array  # index into DecisionBatchPlan.sources
    index: array  # candidate index within the source
    number: array  # how many earlier draws of this record the epoch holds


# build_row(source, record, draw[, segments=..., cached_prompt=...]); the keywords
# are passed only for a candidate with pre-tokenized segments.
RowBuilder = Callable[..., "dict[str, Any] | None"]


class DecisionBatchPlan:
    """Mixture draws, grouping and lazily built canvases for every training epoch."""

    def __init__(
        self,
        pools: Sequence[SourcePool],
        *,
        seed: int,
        micro_batch_size: int,
        stratified: bool,
        weights: Mapping[str, float] | None,
        temperature: float | None,
        limits: BudgetLimits,
        layout: Any,
        build_row: RowBuilder,
    ) -> None:
        if (
            isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or micro_batch_size < 1
        ):
            raise ValueError("decision batch plan needs a positive micro_batch_size")
        self.seed = int(seed)
        self.micro_batch_size = micro_batch_size
        self.stratified = bool(stratified)
        self.limits = limits
        self._layout = layout
        self._build_row = build_row
        self._pools = {pool.source: pool for pool in pools}
        self.budget_drops = {"logical": 0, "physical": 0}
        usable: dict[str, list[int]] = {}
        for source, pool in self._pools.items():
            kept = []
            for index in cap_window(len(pool.entries), pool.cap, self.seed, source):
                entry = pool.entries[index]
                if not isinstance(entry, dict):
                    continue
                reason = budget_reason(entry, limits, layout)
                if reason is None:
                    kept.append(index)
                else:
                    self.budget_drops[reason] += 1
            if kept:
                usable[source] = kept
        if not usable:
            raise ValueError("no decision training rows fit the token budget")
        self._usable = usable
        if isinstance(weights, Mapping):
            weights = {
                source: weights[source] for source in usable if source in weights
            }
        first = DeterministicMixtureSampler(
            usable, weights=weights, temperature=temperature, seed=self.seed
        )
        self.sources = first.sources
        self._source_ids = {source: code for code, source in enumerate(self.sources)}
        self.probabilities = dict(first.probabilities)
        epoch0 = self._draws(
            first.epoch_batches(micro_batch_size, stratified=self.stratified), 0
        )
        self.draws_per_epoch = len(epoch0.index)
        self.batches_per_epoch = self.draws_per_epoch // micro_batch_size
        self._epochs: dict[int, _EpochDraws] = {0: epoch0}
        self.epoch = 0
        self.reserve_records = sum(
            isinstance(entry, bytes)
            for pool in self._pools.values()
            for entry in pool.entries
        )

    # ----------------------------------------------------------------- epochs
    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("decision plan epoch must be a nonnegative integer")
        self.epoch = epoch

    def _draws(self, batches, epoch: int) -> _EpochDraws:
        source_codes, indices, numbers = array("i"), array("q"), array("i")
        seen: dict[tuple[str, int], int] = {}
        for batch in batches:
            if len(batch) != self.micro_batch_size:
                raise ValueError("decision mixture yielded a partial microbatch")
            for sample in batch:
                key = (sample.source, sample.record)
                number = seen.get(key, 0)
                seen[key] = number + 1
                source_codes.append(self._source_ids[sample.source])
                indices.append(sample.record)
                numbers.append(number)
        if not indices:
            raise ValueError("decision epoch plan is empty")
        LOG.debug("Planned decision epoch %s: %s draws", epoch, len(indices))
        return _EpochDraws(source_codes, indices, numbers)

    def epoch_seed(self, epoch: int) -> int:
        return (
            self.seed
            if epoch == 0
            else stable_seed(self.seed, f"decision-epoch:{epoch}")
        )

    def epoch_draws(self, epoch: int) -> _EpochDraws:
        plan = self._epochs.get(epoch)
        if plan is not None:
            return plan
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("decision plan epoch must be a nonnegative integer")
        pools = {}
        for source in self.sources:
            pool = self._pools[source]
            window = [
                index
                for index in cap_window(
                    len(pool.entries), pool.cap, self.seed, source, epoch
                )
                if self._may_fit(pool.entries[index])
            ]
            pools[source] = window or self._usable[source]
        sampler = DeterministicMixtureSampler(
            pools, weights=self.probabilities, seed=self.epoch_seed(epoch)
        )
        plan = self._draws(
            sampler.epoch_batches(
                self.micro_batch_size,
                stratified=self.stratified,
                batch_count=self.batches_per_epoch,
            ),
            epoch,
        )
        if len(self._epochs) >= _EPOCH_PLANS_KEPT:
            for stale in sorted(key for key in self._epochs if key != 0)[:1]:
                del self._epochs[stale]
        self._epochs[epoch] = plan
        return plan

    def _may_fit(self, entry: Any) -> bool:
        if entry is None:
            return False
        if isinstance(entry, bytes):
            return True  # not built yet: checked when drawn
        return budget_reason(entry, self.limits, self._layout) is None

    # ---------------------------------------------------------------- lookups
    def draw(self, epoch: int, position: int) -> tuple[str, int, int]:
        plan = self.epoch_draws(epoch)
        return (
            self.sources[plan.source[position]],
            plan.index[position],
            plan.number[position],
        )

    def epoch_sources(self, epoch: int) -> list[str]:
        plan = self.epoch_draws(epoch)
        return [self.sources[code] for code in plan.source]

    def lengths(self, epoch: int = 0) -> list[int]:
        """Prompt plus canvas tokens per draw (a re-permuted draw may differ by a few)."""
        plan = self.epoch_draws(epoch)
        result = []
        for code, index in zip(plan.source, plan.index, strict=True):
            entry = self._pools[self.sources[code]].entries[index]
            result.append(int(entry["length"]) if isinstance(entry, dict) else 0)
        return result

    def batches(self, epoch: int, *, drop_last: bool = True) -> list[list[int]]:
        """Draw positions per microbatch, in this epoch's (seeded) order."""
        order = random.Random(f"decision-draw-v1:{self.seed}:{epoch}")  # nosec B311
        size = self.micro_batch_size
        if self.stratified:
            groups = [
                list(range(start, start + size))
                for start in range(0, self.draws_per_epoch, size)
            ]
            order.shuffle(groups)
            return groups
        positions = list(range(self.draws_per_epoch))
        order.shuffle(positions)
        return [
            positions[start : start + size]
            for start in range(0, len(positions), size)
            if not drop_last or start + size <= len(positions)
        ]

    def row(self, epoch: int, position: int) -> dict[str, Any]:
        source, index, number = self.draw(epoch, position)
        draw = (epoch, number)
        pool = self._pools[source]
        for candidate in self._candidates(source, index, epoch, position):
            entry = pool.entries[candidate]
            if entry is None:
                continue
            if isinstance(entry, dict) and draw == (0, 0):
                return dict(entry)
            record = (
                pool.records[candidate]
                if isinstance(entry, dict)
                else decode_reserve(entry)
            )
            segments = None if pool.segments is None else pool.segments[candidate]
            if segments is None:
                row = self._build_row(source, record, draw)
            else:
                row = self._build_row(
                    source,
                    record,
                    draw,
                    segments=segments,
                    cached_prompt=entry["canvas"].prompt_ids
                    if isinstance(entry, dict)
                    else None,
                )
            if (
                row is not None
                and budget_reason(row, self.limits, self._layout) is None
            ):
                return row
            if candidate == index:
                LOG.debug(
                    "Decision draw %s/%s (%s #%s) does not fit; using a fallback record",
                    epoch,
                    position,
                    source,
                    index,
                )
        raise RuntimeError(f"no decision row of source {source!r} fits the budget")

    def _candidates(self, source: str, index: int, epoch: int, position: int):
        yield index
        # A drawn record that does not build or fit (only possible for a record not
        # prepared at startup, or a re-permuted one at the budget edge) is replaced by
        # a prepared record of the same source, chosen by the draw's coordinates.
        fallback = self._usable[source]
        start = stable_seed(self.seed, f"decision-fallback:{epoch}:{position}") % len(
            fallback
        )
        for offset in range(len(fallback)):
            yield fallback[(start + offset) % len(fallback)]
