# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Exact pre-optimization outputs and independent scheduling invariants."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from axolotl.utils.samplers.microbatch_balance import balance_microbatches
from axolotl.utils.samplers.rank_balance import order_batches_by_rank

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/sampler_ordering_golden.json").read_text()
)


def digest(value):
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def rank_inputs(case):
    rng = np.random.Generator(np.random.PCG64(case["seed"]))
    costs = (rng.integers(0, 30, case["dp"] * case["gas"] * 5) * 64).tolist()
    return [256] * len(costs) if case["seed"] == 0 else costs


def refinement_inputs(case):
    rng = np.random.Generator(np.random.PCG64(case["seed"]))
    batches = []
    index = 0
    for _ in range(case["width"] * 32 + 3):
        if case["mode"] == "padded":
            sizes = [1] * 4
        elif case["mode"] == "flattened":
            sizes = [4]
        else:
            sizes = rng.integers(1, 5, size=int(rng.integers(1, 3))).tolist()
        rows = []
        for size in sizes:
            rows.append(list(range(index, index + size)))
            index += size
        batches.append(rows)
    lengths = rng.integers(2, 401, index)
    counts = np.array([rng.integers(0, n) for n in lengths])
    starts = np.minimum(counts, rng.integers(0, 2, index))
    if case["mode"] == "padded":
        starts[:] = 0
    capacity = (
        int(max(sum(lengths[row]) for batch in batches for row in batch))
        if case["mode"] == "packed"
        else None
    )
    return dict(
        batches=batches,
        lengths=lengths.tolist(),
        counts=counts.tolist(),
        starts=starts.tolist(),
        kwargs=dict(padding_multiple=64, capacity=capacity),
    )


@pytest.mark.parametrize("case", GOLDEN["rank"])
def test_rank_order_matches_reference(case):
    inputs = rank_inputs(case)
    assert digest(inputs) == case["inputs_sha256"], "Seeded inputs changed"
    costs = np.array(inputs)
    options = dict(dp=case["dp"], gas=case["gas"], window_steps=3, beam_width=3)
    order = order_batches_by_rank(costs, **options)
    assert digest(order.tolist()) == case["expected_sha256"], "Rank order changed"
    np.testing.assert_array_equal(order, order_batches_by_rank(costs, **options))
    assert costs.tolist() == inputs
    width = case["dp"] * case["gas"]
    for offset in range(0, len(order), width):
        assert sorted(order[offset : offset + width]) == list(
            range(offset, offset + width)
        )
    before = costs.reshape(-1, case["gas"], case["dp"]).astype(float)
    after = costs[order].reshape(before.shape)
    assert np.all(after.max(2).sum(1) <= before.max(2).sum(1))
    assert np.all(
        ((after - after.mean(2, keepdims=True)) ** 2).sum((1, 2))
        <= ((before - before.mean(2, keepdims=True)) ** 2).sum((1, 2)) + 1e-6
    )


@pytest.mark.parametrize("case", GOLDEN["refine"])
def test_refinement_matches_reference(case):
    inputs = refinement_inputs(case)
    assert digest(inputs) == case["inputs_sha256"], "Seeded inputs changed"
    batches = deepcopy(inputs["batches"])
    args = [np.array(inputs[name]) for name in ["lengths", "counts", "starts"]]
    width = case["width"]
    within = balance_microbatches(batches, *args, width, **inputs["kwargs"])
    wide = balance_microbatches(
        batches, *args, width, window_steps=32, **inputs["kwargs"]
    )
    combined = balance_microbatches(wide, *args, width, **inputs["kwargs"])
    assert digest(within) == case["within_sha256"], "within ordering changed"
    assert digest(wide) == case["wide_sha256"], "wide ordering changed"
    assert digest(combined) == case["combined_sha256"], "combined ordering changed"
    assert batches == inputs["batches"]
    for i in range(0, len(wide), width):
        assert Counter(
            j for b in wide[i : i + width] for row in b for j in row
        ) == Counter(j for b in combined[i : i + width] for row in b for j in row)


@pytest.mark.parametrize(
    "options",
    [{"dp": 0}, {"dp": 5}, {"gas": 0}, {"window_steps": 0}, {"beam_width": 0}],
)
def test_invalid_rank_dimensions(options):
    with pytest.raises(ValueError):
        order_batches_by_rank(np.ones(16), **options)


@pytest.mark.parametrize(
    "costs", [[-1] * 16, [float("nan")] * 16, [float("inf")] * 16, [[1] * 16], [1] * 15]
)
def test_invalid_rank_costs(costs):
    with pytest.raises(ValueError):
        order_batches_by_rank(costs)


def test_empty_rank_order():
    assert order_batches_by_rank([]).tolist() == []
