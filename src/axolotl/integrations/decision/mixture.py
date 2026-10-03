"""Deterministic, decontaminated source mixing for decision training."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .hygiene import decontaminate

T = TypeVar("T")


@dataclass(frozen=True)
class PreparedSourcePools:
    """Decontaminated and capped normalized records partitioned by source."""

    records: Mapping[str, tuple[Mapping[str, Any], ...]]
    counts: Mapping[str, int]


@dataclass(frozen=True)
class MixtureSample(Generic[T]):
    """A training example with its source retained for metrics and loss weighting."""

    source: str
    record: T


class _SourceCursor(Generic[T]):
    def __init__(self, records: Sequence[T], seed: int) -> None:
        self._records = tuple(records)
        self._order = list(range(len(records)))
        self._random = random.Random(seed)  # nosec B311 - Deterministic training order.
        self._random.shuffle(self._order)
        self._offset = 0

    def next(self) -> T:
        if self._offset == len(self._order):
            self._random.shuffle(self._order)
            self._offset = 0
        record = self._records[self._order[self._offset]]
        self._offset += 1
        return record


class DeterministicMixtureSampler(Generic[T]):
    """Draw source-tagged examples and source-stratified microbatches."""

    def __init__(
        self,
        pools: Mapping[str, Sequence[T]],
        *,
        weights: Mapping[str, float] | None = None,
        temperature: float | None = None,
        seed: int = 0,
    ) -> None:
        self._pools = _validate_pools(pools)
        self.sources = tuple(sorted(self._pools))
        self.probabilities = _probabilities(
            self._pools, self.sources, weights, temperature
        )
        self._probability_values = tuple(
            self.probabilities[source] for source in self.sources
        )
        self._seed = seed
        self._source_random = random.Random(seed)  # nosec B311 - Deterministic draws.
        self._cursors = {
            source: _SourceCursor(records, _source_seed(seed, source))
            for source, records in self._pools.items()
        }
        self._fractional_credit = {source: 0.0 for source in self.sources}

    def draw(self, count: int) -> tuple[MixtureSample[T], ...]:
        """Draw examples by source probability, cycling deterministically within sources."""
        _validate_count(count, "count")
        choices = self._source_random.choices(
            self.sources, weights=self._probability_values, k=count
        )
        return tuple(
            MixtureSample(source, self._cursors[source].next()) for source in choices
        )

    def batches(
        self, batch_size: int, count: int, *, stratified: bool = True
    ) -> Iterator[tuple[MixtureSample[T], ...]]:
        """Yield fixed-size deterministic microbatches without flattening examples."""
        _validate_count(batch_size, "batch_size", positive=True)
        _validate_count(count, "count")
        for _ in range(count):
            if stratified:
                sources = self._stratified_sources(batch_size)
                self._source_random.shuffle(sources)
                yield tuple(
                    MixtureSample(source, self._cursors[source].next())
                    for source in sources
                )
            else:
                yield self.draw(batch_size)

    def epoch_batches(
        self, batch_size: int, *, stratified: bool = True
    ) -> Iterator[tuple[MixtureSample[T], ...]]:
        """Yield batches until every capped source record is seen once."""
        _validate_count(batch_size, "batch_size", positive=True)
        epoch_seed = self._source_random.randrange(1 << 63)  # nosec B311 - Reproducible seed.
        cursors = {
            source: _SourceCursor(records, _source_seed(epoch_seed, source))
            for source, records in self._pools.items()
        }
        credit = {source: 0.0 for source in self.sources}
        seen = {source: 0 for source in self.sources}
        while not all(
            seen[source] >= len(self._pools[source]) for source in self.sources
        ):
            if stratified:
                sources = self._stratified_sources(batch_size, credit)
                random.Random(epoch_seed).shuffle(sources)  # nosec B311 - Reproducible order.
            else:
                sources = self._source_random.choices(
                    self.sources, weights=self._probability_values, k=batch_size
                )
            epoch_seed += 1
            batch = tuple(
                MixtureSample(source, cursors[source].next()) for source in sources
            )
            for source in sources:
                seen[source] += 1
            yield batch

    def _stratified_sources(
        self, batch_size: int, fractional_credit: dict[str, float] | None = None
    ) -> list[str]:
        credit = (
            self._fractional_credit if fractional_credit is None else fractional_credit
        )
        expected = {
            source: self.probabilities[source] * batch_size for source in self.sources
        }
        fixed = {source: math.floor(value) for source, value in expected.items()}
        remaining = batch_size - sum(fixed.values())
        selected = [source for source in self.sources for _ in range(fixed[source])]
        for source in self.sources:
            credit[source] += expected[source] - fixed[source]
        for _ in range(remaining):
            source = max(
                enumerate(self.sources),
                key=lambda item: (credit[item[1]], -item[0]),
            )[1]
            selected.append(source)
            credit[source] -= 1.0
        return selected


def prepare_source_pools(
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    *,
    max_examples_per_source: int | Mapping[str, int] | None = None,
    exclude_families: bool = True,
    seed: int = 0,
) -> PreparedSourcePools:
    """Decontaminate records, then deterministically cap each nonempty source pool."""
    retained, counts = decontaminate(
        training, evaluation, exclude_families=exclude_families
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in retained:
        grouped[record["source"]].append(record)
    _validate_caps(max_examples_per_source, grouped)
    pools = {
        source: _cap_records(
            records, _cap_for_source(max_examples_per_source, source), seed
        )
        for source, records in grouped.items()
    }
    records = {source: pool for source, pool in pools.items() if pool}
    if not records:
        raise ValueError("no nonempty source pools remain after decontamination")
    capped = sum(len(grouped[source]) - len(pool) for source, pool in records.items())
    result_counts = dict(counts)
    result_counts["capped"] = capped
    result_counts["sources"] = len(records)
    return PreparedSourcePools(records=records, counts=result_counts)


def source_counts(samples: Sequence[MixtureSample[object]]) -> Mapping[str, int]:
    """Count source-tagged samples without inspecting or altering their records."""
    return dict(Counter(sample.source for sample in samples))


def _validate_pools(pools: Mapping[str, Sequence[T]]) -> dict[str, tuple[T, ...]]:
    if not pools:
        raise ValueError("mixture requires at least one source pool")
    result = {}
    for source, records in pools.items():
        if not isinstance(source, str) or not source:
            raise ValueError("mixture source names must be nonempty strings")
        if not records:
            raise ValueError(f"mixture source {source!r} is empty")
        result[source] = tuple(records)
    return result


def _probabilities(
    pools: Mapping[str, Sequence[object]],
    sources: tuple[str, ...],
    weights: Mapping[str, float] | None,
    temperature: float | None,
) -> Mapping[str, float]:
    if weights is not None:
        if temperature is not None:
            raise ValueError("specify explicit weights or temperature, not both")
        if set(weights) != set(sources):
            raise ValueError("explicit mixture weights must name every nonempty source")
        values = {
            source: _positive_finite(weights[source], "mixture weight")
            for source in sources
        }
        return _normalize_probabilities(values)
    else:
        resolved_temperature = 0.5 if temperature is None else temperature
        exponent = _nonnegative_finite(resolved_temperature, "mixture temperature")
        log_values = {
            source: exponent * math.log(len(pools[source])) for source in sources
        }
        if any(not math.isfinite(value) for value in log_values.values()):
            raise ValueError("mixture temperature produces nonfinite source weights")
        maximum = max(log_values.values())
        values = {
            source: math.exp(value - maximum) for source, value in log_values.items()
        }
        return _normalize_probabilities(values)


def _normalize_probabilities(values: Mapping[str, float]) -> Mapping[str, float]:
    maximum = max(values.values())
    scaled = {source: value / maximum for source, value in values.items()}
    total = sum(scaled.values())
    probabilities = {source: value / total for source, value in scaled.items()}
    if any(not math.isfinite(value) or value <= 0 for value in probabilities.values()):
        raise ValueError("mixture source probability underflowed to zero")
    return probabilities


def _validate_caps(
    cap: int | Mapping[str, int] | None, pools: Mapping[str, Sequence[object]]
) -> None:
    if isinstance(cap, Mapping):
        unknown = set(cap) - set(pools)
        if unknown:
            raise ValueError("per-source cap names an unavailable source")
        for value in cap.values():
            _positive_integer(value, "max_examples_per_source")
    elif cap is not None:
        _positive_integer(cap, "max_examples_per_source")


def _cap_for_source(cap: int | Mapping[str, int] | None, source: str) -> int | None:
    if isinstance(cap, Mapping):
        return cap.get(source)
    return cap


def _cap_records(
    records: Sequence[Mapping[str, Any]], cap: int | None, seed: int
) -> tuple[Mapping[str, Any], ...]:
    if cap is None or len(records) <= cap:
        return tuple(records)
    order = list(range(len(records)))
    random.Random(_source_seed(seed, records[0]["source"])).shuffle(  # nosec B311
        order
    )
    return tuple(records[index] for index in order[:cap])


def _source_seed(seed: int, source: str) -> int:
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be nonnegative and finite")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_count(value: int, name: str, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (value < 0 or (positive and value == 0))
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
