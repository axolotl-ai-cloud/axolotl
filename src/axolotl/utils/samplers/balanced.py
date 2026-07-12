"""Balanced sample-packing helpers."""

from __future__ import annotations

import heapq
import math
from typing import Literal

import numpy as np

PackingStrategy = Literal["first_fit_decreasing", "balanced_greedy"]
FIRST_FIT_DECREASING: PackingStrategy = "first_fit_decreasing"
BALANCED_GREEDY: PackingStrategy = "balanced_greedy"


def default_sample_packing_strategy(is_multimodal: bool) -> PackingStrategy:
    return BALANCED_GREEDY if is_multimodal else FIRST_FIT_DECREASING


def balanced_greedy_pack_group(
    sequence_lengths: np.ndarray,
    group_offset: int,
    bin_capacity: int,
    bin_size: int,
) -> list[list[int]]:
    if bin_capacity <= 0:
        raise ValueError("bin_capacity must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    items = sorted(
        enumerate(sequence_lengths),
        key=lambda item: int(item[1]),
        reverse=True,
    )
    if not items:
        return []

    total_length = sum(max(0, int(length)) for _, length in items)
    min_bins = max(1, math.ceil(total_length / bin_capacity))
    bins: list[list[int]] = [[] for _ in range(min_bins)]
    bin_lengths = [0 for _ in range(min_bins)]
    available = [(0, idx) for idx in range(min_bins)]
    heapq.heapify(available)

    # Invariant: each non-full bin has exactly one live heap entry, so the
    # least-used bin is always at the top and entries are never stale.
    for seq_id, raw_length in items:
        length = int(raw_length)
        global_idx = seq_id + group_offset
        placed = False

        if available:
            used, bin_idx = heapq.heappop(available)
            if used + length <= bin_capacity:
                bins[bin_idx].append(global_idx)
                bin_lengths[bin_idx] += length
                if len(bins[bin_idx]) < bin_size:
                    heapq.heappush(available, (bin_lengths[bin_idx], bin_idx))
                placed = True
            else:
                # Bins share one `bin_capacity`; if the least-used bin can't fit
                # this item, no other bin can either.
                heapq.heappush(available, (used, bin_idx))

        if not placed:
            bins.append([global_idx])
            bin_lengths.append(length)
            if bin_size > 1:
                heapq.heappush(available, (length, len(bins) - 1))

    return [packed for packed in bins if packed]
