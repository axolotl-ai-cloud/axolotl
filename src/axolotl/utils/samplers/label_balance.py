# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded label balancing that preserves packed capacity and sample coverage."""

import heapq

import numpy as np


def _bin_labels(bin_, counts, starts):
    return sum(int(counts[i]) for i in bin_) - int(starts[bin_[0]])


def _batch_labels(batch, counts, starts):
    return sum(_bin_labels(bin_, counts, starts) for bin_ in batch)


def _padded_slots(batches, lengths, multiple):
    return sum(
        (
            (max(sum(int(lengths[i]) for i in bin_) for bin_ in batch) + multiple - 1)
            // multiple
        )
        * multiple
        * len(batch)
        for batch in batches
    )


def _regroup(batches, counts, starts):
    bins = [bin_ for batch in batches for bin_ in batch]
    bins.sort(key=lambda bin_: _bin_labels(bin_, counts, starts), reverse=True)
    result: list[list[list[int]]] = [[] for _ in batches]
    heap = [(0, i) for i in range(len(batches))]
    for bin_ in bins:
        total, i = heapq.heappop(heap)
        result[i].append(bin_)
        if len(result[i]) < len(batches[i]):
            heapq.heappush(heap, (total + _bin_labels(bin_, counts, starts), i))
    return result


def _swap(high, low, lengths, counts, starts, capacity, difference, padding_multiple):
    high_lengths = [sum(int(lengths[i]) for i in bin_) for bin_ in high]
    low_lengths = [sum(int(lengths[i]) for i in bin_) for bin_ in low]
    high_items = [(b, p, i) for b, bin_ in enumerate(high) for p, i in enumerate(bin_)]
    low_items = [(b, p, i) for b, bin_ in enumerate(low) for p, i in enumerate(bin_)]
    # Bound candidate work even when a row packs thousands of short samples.
    high_items.sort(key=lambda item: int(counts[item[2]]), reverse=True)
    low_items.sort(key=lambda item: int(counts[item[2]]))
    padding_budget = (
        _padded_slots([high, low], lengths, padding_multiple)
        if padding_multiple is not None
        else None
    )
    best = None
    best_gain = 0
    for hb, hp, a in high_items[:32]:
        for lb, lp, b in low_items[:32]:
            length_delta = int(lengths[a]) - int(lengths[b])
            if (
                high_lengths[hb] - length_delta > capacity
                or low_lengths[lb] + length_delta > capacity
            ):
                continue
            start_delta = int(starts[a]) - int(starts[b])
            transfer = int(counts[a]) - int(counts[b])
            high_transfer = transfer - (start_delta if hp == 0 else 0)
            low_transfer = transfer - (start_delta if lp == 0 else 0)
            # Preserve the loss denominator when swapping row-leading samples.
            if high_transfer != low_transfer:
                continue
            gain = 2 * high_transfer * (difference - high_transfer)
            if gain > best_gain:
                if padding_multiple is not None:
                    high_used, low_used = high_lengths.copy(), low_lengths.copy()
                    high_used[hb] -= length_delta
                    low_used[lb] += length_delta
                    slots = sum(
                        ((max(used) + padding_multiple - 1) // padding_multiple)
                        * padding_multiple
                        * len(used)
                        for used in (high_used, low_used)
                    )
                    if slots > padding_budget:
                        continue
                best_gain = gain
                best = (hb, hp, lb, lp)
    if best is None:
        return False
    hb, hp, lb, lp = best
    high[hb][hp], low[lb][lp] = low[lb][lp], high[hb][hp]
    return True


def _repack(high, low, lengths, counts, starts, capacity, bin_size, padding_multiple):
    items = [i for batch in (high, low) for bin_ in batch for i in bin_]
    items.sort(key=lambda i: (int(lengths[i]), int(counts[i])), reverse=True)
    result: list[list[list[int]]] = [[[] for _ in batch] for batch in (high, low)]
    used = [[0 for _ in batch] for batch in (high, low)]
    totals = [0, 0]
    for i in items:
        best = None
        best_score = None
        for batch_idx, batch in enumerate(result):
            for bin_idx, bin_ in enumerate(batch):
                remaining = capacity - used[batch_idx][bin_idx] - int(lengths[i])
                if remaining < 0 or len(bin_) >= bin_size:
                    continue
                score = (totals[batch_idx], remaining, batch_idx, bin_idx)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (batch_idx, bin_idx)
        if best is None:
            return None
        batch_idx, bin_idx = best
        bin_ = result[batch_idx][bin_idx]
        totals[batch_idx] += int(counts[i]) - (int(starts[i]) if not bin_ else 0)
        bin_.append(i)
        used[batch_idx][bin_idx] += int(lengths[i])
    if any(not bin_ for batch in result for bin_ in batch):
        return None
    if padding_multiple is not None and _padded_slots(
        result, lengths, padding_multiple
    ) > _padded_slots([high, low], lengths, padding_multiple):
        return None
    old = [_batch_labels(batch, counts, starts) for batch in (high, low)]
    if sum(totals) == sum(old) and abs(totals[0] - totals[1]) < abs(old[0] - old[1]):
        return result
    return None


def balance_labels(
    batches: list[list[list[int]]],
    lengths: np.ndarray,
    counts: np.ndarray,
    starts: np.ndarray,
    capacity: int,
    bin_size: int,
    seed: int,
    window_size: int = 64,
    padding_multiple: int | None = None,
    batches_per_optimizer_step: int = 1,
) -> list[list[list[int]]]:
    """Reduce full-batch label variance without adding bins or changing the tail.

    Counts include all unmasked labels. Starts identifies labels excluded by causal
    shifting at the beginning of each packed row. Windows bound the search to four
    passes of paired swaps/repacking; unsuccessful candidates leave packing intact.
    When padding_multiple is provided, total padded tensor size cannot increase.
    """
    if batches_per_optimizer_step < 1:
        raise ValueError("batches_per_optimizer_step must be positive")
    if padding_multiple is not None and padding_multiple <= 0:
        raise ValueError("padding_multiple must be positive")
    if not batches:
        return batches
    result = []
    full_count = len(batches)
    if len(batches[-1]) < len(batches[0]):
        full_count -= 1
    full_count -= full_count % batches_per_optimizer_step
    rng = np.random.default_rng(seed)
    for offset in range(0, full_count, window_size):
        window = [
            [list(bin_) for bin_ in batch]
            for batch in batches[offset : min(offset + window_size, full_count)]
        ]
        totals = [_batch_labels(batch, counts, starts) for batch in window]
        candidate = _regroup(window, counts, starts)
        candidate_totals = [_batch_labels(batch, counts, starts) for batch in candidate]
        if sum(x * x for x in candidate_totals) < sum(x * x for x in totals) and (
            padding_multiple is None
            or _padded_slots(candidate, lengths, padding_multiple)
            <= _padded_slots(window, lengths, padding_multiple)
        ):
            window, totals = candidate, candidate_totals
        for _ in range(4):
            order = sorted(range(len(window)), key=lambda i: totals[i])
            changed = False
            for lo, hi in zip(order[: len(order) // 2], reversed(order), strict=False):
                difference = totals[hi] - totals[lo]
                if difference <= 1:
                    continue
                improved = _swap(
                    window[hi],
                    window[lo],
                    lengths,
                    counts,
                    starts,
                    capacity,
                    difference,
                    padding_multiple,
                )
                if not improved:
                    replacement = _repack(
                        window[hi],
                        window[lo],
                        lengths,
                        counts,
                        starts,
                        capacity,
                        bin_size,
                        padding_multiple,
                    )
                    if replacement is None:
                        continue
                    window[hi], window[lo] = replacement
                totals[hi] = _batch_labels(window[hi], counts, starts)
                totals[lo] = _batch_labels(window[lo], counts, starts)
                changed = True
            if not changed:
                break
        rng.shuffle(window)
        result.extend(window)
    result.extend(batches[full_count:])
    return result
