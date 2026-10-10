# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Configurable cross-step refinement and deterministic sampler replay."""

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
from torch.utils.data import RandomSampler

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.samplers import LabelBalancedRandomSampler, MultipackBatchSampler
from axolotl.utils.samplers.microbatch_balance import balance_microbatches
from axolotl.utils.schemas.config import AxolotlInputConfig


def make_sampler(mode, window):
    rng = np.random.default_rng(42)
    lengths = rng.integers(2, 33, 1027)
    counts = np.array([rng.integers(1, n) for n in lengths])
    kwargs = dict(
        lengths=lengths,
        label_counts=counts,
        seed=42,
        batches_per_optimizer_step=4,
        dp_count=2,
        label_balance_window_optim_steps=window,
    )
    if mode == "packed":
        return MultipackBatchSampler(
            RandomSampler(range(len(lengths))),
            batch_size=1,
            batch_max_len=128,
            bin_size=16,
            num_processes=1,
            **kwargs,
        )
    return LabelBalancedRandomSampler(batch_size=4, length_mode=mode, **kwargs)


@pytest.mark.parametrize("mode", ["packed", "padded", "flattened"])
@pytest.mark.parametrize("window", [1, 8, 32])
def test_window_pipeline_and_protected_step_variance(mode, window):
    sampler = make_sampler(mode, window)
    calls = []

    def checked(batches, lengths, counts, starts, step_width, **kwargs):
        after = balance_microbatches(
            batches, lengths, counts, starts, step_width, **kwargs
        )
        width = step_width * kwargs["window_steps"]
        calls.append(kwargs["window_steps"])
        end = len(batches) // width * width
        assert after[end:] == batches[end:]
        for offset in range(0, end, width):
            before_window = batches[offset : offset + width]
            after_window = after[offset : offset + width]
            assert Counter(i for b in before_window for r in b for i in r) == Counter(
                i for b in after_window for r in b for i in r
            )
            for values, shift in [(lengths, np.zeros_like(starts)), (counts, starts)]:

                def totals(rows, values=values, shift=shift):
                    return (
                        np.array(
                            [
                                sum(
                                    sum(int(values[i]) for i in row)
                                    - int(shift[row[0]])
                                    for row in batch
                                )
                                for batch in rows
                            ]
                        )
                        .reshape(-1, step_width)
                        .sum(1)
                    )

                a, b = totals(before_window), totals(after_window)
                assert a.sum() == b.sum()
                assert np.var(b) <= np.var(a) + 1e-8
        return after

    module = "multipack" if mode == "packed" else "label_balanced"
    with patch(f"axolotl.utils.samplers.{module}.balance_microbatches", checked):
        result = sampler.generate_batches() if mode == "packed" else list(sampler)
    assert calls == ([1] if window == 1 else [window, 1])
    replay = make_sampler(mode, window)
    other = replay.generate_batches() if mode == "packed" else list(replay)
    assert result == other
    settings = AxolotlTrainer._balanced_sampler_settings(sampler)
    if window == 1:
        assert "label_balance_window_optim_steps" not in settings
    else:
        assert settings["label_balance_window_optim_steps"] == window


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
@pytest.mark.parametrize("mode", ["packed", "padded", "flattened"])
def test_sampler_rejects_invalid_window(mode, window):
    with pytest.raises(ValueError, match="positive integer"):
        make_sampler(mode, window)


@pytest.mark.parametrize("window", [1, 8, 32])
def test_config_window(window):
    cfg = AxolotlInputConfig(
        base_model="test",
        datasets=[{"path": "test", "type": "alpaca"}],
        learning_rate=1e-5,
        balance_labels=True,
        label_balance_window_optim_steps=window,
    )
    assert cfg.label_balance_window_optim_steps == window


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_config_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        AxolotlInputConfig(
            base_model="test",
            datasets=[{"path": "test", "type": "alpaca"}],
            learning_rate=1e-5,
            label_balance_window_optim_steps=window,
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"balance_labels": False}, "requires balance_labels"),
        (
            {"balance_labels": True, "sample_packing": True, "streaming": True},
            "non-streaming",
        ),
    ],
)
def test_config_rejects_inactive_or_streaming_window(options, match):
    with pytest.raises(ValueError, match=match):
        AxolotlInputConfig(
            base_model="test",
            datasets=[{"path": "test", "type": "alpaca"}],
            learning_rate=1e-5,
            max_steps=8,
            label_balance_window_optim_steps=8,
            **options,
        )


def test_cross_step_window_improves_totals_that_within_step_cannot_change():
    lengths = np.array([9, 9, 1, 1, 8])
    counts = lengths.copy()
    starts = np.zeros(5, dtype=int)
    batches = [[[0, 1]], [[2, 3]], [[4]]]
    within = balance_microbatches(batches, lengths, counts, starts, 1)
    assert within == batches
    cross = balance_microbatches(batches, lengths, counts, starts, 1, window_steps=2)
    assert cross[-1] == batches[-1]
    assert [sum(lengths[i] for i in b[0]) for b in cross[:2]] == [10, 10]
