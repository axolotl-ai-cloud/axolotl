# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Deterministic bounded search for rank-local batch ordering."""

import itertools
from numbers import Integral

import numpy as np


def order_batches_by_rank(costs, dp=4, gas=4, window_steps=32, beam_width=8):
    """Return batch indices ordered by rank while preserving step membership.

    Costs describe rank-local padded or flattened tensor sizes. Input order is
    microstep-major, then rank. Search is exhaustive over rank permutations for
    up to four ranks, and uses at most 24 deterministic permutations above that.
    Incomplete optimizer steps are rejected.
    """
    if any(
        not isinstance(n, Integral) or isinstance(n, bool) or n < 1
        for n in (dp, gas, window_steps, beam_width)
    ):
        raise ValueError(
            "Ranks, GAS, window, and beam widths must be positive integers"
        )
    costs = np.asarray(costs)
    if costs.ndim != 1 or np.any(costs < 0) or not np.all(np.isfinite(costs)):
        raise ValueError("Costs must be finite, nonnegative, and one-dimensional")
    if not len(costs):
        return np.empty(0, dtype=np.int64)
    if dp <= 4:
        permutations = np.array(list(itertools.permutations(range(dp))))
    else:
        ranks = np.arange(dp)
        shifts = np.unique(np.linspace(0, dp - 1, min(dp, 12), dtype=int))
        permutations = np.array(
            [
                np.roll(ranks[::direction], shift)
                for direction in (1, -1)
                for shift in shifts
            ]
        )
    width = dp * gas
    if len(costs) % width:
        raise ValueError("Complete optimizer steps required")
    order = []
    energy = np.zeros(dp, dtype=float)
    work = np.zeros(dp, dtype=float)
    last = None
    cumulative_spread = 0.0
    for window in range(0, len(costs), width * window_steps):
        beam = [(energy.copy(), work.copy(), last, cumulative_spread, [])]
        for start in range(
            window, min(len(costs), window + width * window_steps), width
        ):
            ids = np.arange(start, start + width)
            sorted_ids = ids[np.argsort(costs[ids], kind="stable")]
            layouts = [
                ids.reshape(gas, dp),
                sorted_ids.reshape(gas, dp),
                sorted_ids.reshape(dp, gas).T,
            ]
            snake = sorted_ids.reshape(gas, dp).copy()
            snake[1::2] = snake[1::2, ::-1]
            layouts.append(snake)
            original = costs[ids].reshape(gas, dp).astype(float)
            spread_cap = float(
                ((original - original.mean(1, keepdims=True)) ** 2).sum()
            )
            max_sum_cap = float(original.max(1).sum())
            candidates = np.concatenate(
                [
                    base[:, permutations].transpose(1, 0, 2)
                    for layout in layouts
                    for base in (layout, layout[::-1])
                ]
            )
            _, first_occurrence = np.unique(
                candidates.reshape(len(candidates), -1), axis=0, return_index=True
            )
            candidates = candidates[np.sort(first_occurrence)]
            values = costs[candidates].astype(float)
            spread = ((values - values.mean(axis=2, keepdims=True)) ** 2).sum(
                axis=(1, 2)
            )
            valid = (spread <= spread_cap + 1e-6) & (
                values.max(axis=2).sum(axis=1) <= max_sum_cap + 1e-6
            )
            candidates = candidates[valid]
            values = values[valid]
            spread = spread[valid]
            first = values[:, 0, :]
            final = values[:, -1, :]
            delta = np.diff(values, axis=1)
            internal = (delta * delta).sum(axis=1)
            added = values.sum(axis=1)
            energies = np.array([item[0] for item in beam])[:, None, :] + internal
            if beam[0][2] is not None:
                previous = np.array([item[2] for item in beam])[:, None, :]
                energies += (first - previous) ** 2
            workloads = np.array([item[1] for item in beam])[:, None, :] + added
            spreads = np.array([item[3] for item in beam])[:, None] + spread
            worst = energies.max(axis=2)
            imbalance = ((workloads - workloads.mean(axis=2, keepdims=True)) ** 2).sum(
                axis=2
            )
            total = energies.sum(axis=2)
            # Stable ties retain the reference beam-major, candidate-major order.
            chosen = np.lexsort(
                (total.ravel(), spreads.ravel(), imbalance.ravel(), worst.ravel())
            )[:beam_width]
            updated = []
            for index in chosen:
                parent, candidate = divmod(int(index), len(candidates))
                updated.append(
                    (
                        energies[parent, candidate],
                        workloads[parent, candidate],
                        final[candidate],
                        spreads[parent, candidate],
                        beam[parent][4] + [candidates[candidate]],
                    )
                )
            beam = updated
        energy, work, last, cumulative_spread, path = beam[0]
        order.extend(np.concatenate(path).ravel().tolist())
    return np.array(order, dtype=int)
