# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Production sampler boundaries, distributed ordering, and deterministic replay."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset
from torch.utils.data import BatchSampler, RandomSampler

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.samplers import LabelBalancedRandomSampler, MultipackBatchSampler
from axolotl.utils.samplers.rank_balance import order_batches_by_rank


@pytest.mark.parametrize("dp", [5, 8, 16, 64])
@pytest.mark.parametrize("gas", [1, 4])
def test_large_rank_order_invariants(dp, gas):
    costs = np.random.default_rng(8).integers(1, 1000, dp * gas * 7)
    order = order_batches_by_rank(costs, dp=dp, gas=gas, window_steps=3)
    np.testing.assert_array_equal(
        order, order_batches_by_rank(costs, dp=dp, gas=gas, window_steps=3)
    )
    for start in range(0, len(costs), dp * gas):
        assert sorted(order[start : start + dp * gas]) == list(
            range(start, start + dp * gas)
        )
    before = costs.reshape(-1, gas, dp).astype(float)
    after = costs[order].reshape(before.shape)
    assert np.all(after.max(2).sum(1) <= before.max(2).sum(1))
    assert np.all(
        ((after - after.mean(2, keepdims=True)) ** 2).sum((1, 2))
        <= ((before - before.mean(2, keepdims=True)) ** 2).sum((1, 2)) + 1e-6
    )


def packed_sampler(dp=1, seed=42, count=259):
    lengths = np.random.default_rng(7).integers(2, 16, count)
    return MultipackBatchSampler(
        RandomSampler(range(count)),
        batch_size=1,
        batch_max_len=32,
        lengths=lengths,
        label_counts=lengths // 2,
        bin_size=8,
        num_processes=1,
        seed=seed,
        dp_count=dp,
        batches_per_optimizer_step=4 * dp,
    )


def test_length_estimate_reuses_deterministic_balanced_plan():
    import axolotl.utils.samplers.multipack as module

    sampler = packed_sampler()
    with patch.object(module, "balance_labels", wraps=module.balance_labels) as balance:
        size = len(sampler)
        expected = sampler.generate_batches()
        assert len(list(sampler)) == size
        assert list(sampler) == expected
        assert balance.call_count == 1
        sampler.set_epoch(1)
        len(sampler)
        assert balance.call_count == 2
        assert sampler.label_metrics is not None


@pytest.mark.parametrize("mode", ["packed", "padded", "flattened"])
@pytest.mark.parametrize("dp", [2, 8])
def test_rank_order_is_integrated_and_preserves_step_membership(mode, dp):
    if mode == "packed":
        sampler = packed_sampler(dp)
        import axolotl.utils.samplers.multipack as module

        def batches():
            return list(sampler.generate_batches())
    else:
        lengths = np.random.default_rng(9).integers(2, 32, 4 * dp * 4 * 3 + 7)
        sampler = LabelBalancedRandomSampler(
            lengths,
            lengths // 2,
            4,
            dp_count=dp,
            batches_per_optimizer_step=dp * 4,
            length_mode=mode,
        )
        import axolotl.utils.samplers.label_balanced as module

        def batches():
            return [[batch] for batch in BatchSampler(sampler, 4, False)]

    with patch.object(
        module,
        "order_batches_by_rank",
        side_effect=lambda costs, **kw: np.arange(len(costs)),
    ):
        before = batches()
    sampler.set_epoch(0)
    with patch.object(
        module, "order_batches_by_rank", wraps=order_batches_by_rank
    ) as ordering:
        after = batches()
        assert ordering.call_count == 1
        assert ordering.call_args.kwargs == {"dp": dp, "gas": 4}
    cutoff = len(before) // (dp * 4) * (dp * 4)
    assert before[cutoff:] == after[cutoff:]
    for i in range(0, cutoff, dp * 4):

        def members(rows):
            return Counter(tuple(tuple(row) for row in batch) for batch in rows)

        assert members(before[i : i + dp * 4]) == members(after[i : i + dp * 4])
    assert sampler.label_metrics["after"] == pytest.approx(
        sampler.label_metrics["before_rank"]
    )


@pytest.mark.parametrize("seed", [None, 0, 7])
@pytest.mark.parametrize("window", [1, 8])
def test_packing_uses_effective_batch_size_and_data_seed(seed, window):
    trainer = object.__new__(AxolotlTrainer)
    trainer._train_batch_size = 3
    trainer.state = SimpleNamespace(train_batch_size=99)
    trainer.args = SimpleNamespace(
        multipack_real_batches=False,
        per_device_train_batch_size=2,
        max_seq_length=8,
        balance_labels=True,
        seed=42,
        data_seed=seed,
        label_balance_window_optim_steps=window,
        world_size=2,
        gradient_accumulation_steps=4,
        sample_packing_efficiency=1,
        sample_packing_group_size=100,
        sample_packing_bin_size=8,
        sample_packing_sequentially=False,
        dataset_num_proc=1,
        sample_packing_mp_start_method="fork",
    )
    data = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * 32, "labels": [[1, 2, 3, 4]] * 32}
    )
    sampler = trainer._create_multipack_sampler(RandomSampler(data), data)
    assert sampler.label_balance_window_optim_steps == window
    assert sampler.batch_max_len == 24
    assert sampler.seed == (42 if seed is None else seed)
    assert sampler.dp_count == 2


@pytest.mark.parametrize("offset", [1, 3.99, 4.1, -4, 17])
def test_resume_rejects_invalid_update_boundary(offset):
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
    with pytest.raises(ValueError, match="optimizer-step boundary"):
        trainer._validated_balanced_offset(offset, 16)


def test_resume_accepts_float_roundoff_and_final_partial_update():
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
    assert trainer._validated_balanced_offset(4 + 1e-10, 15) == 4
    assert trainer._validated_balanced_offset(15, 15) == 15


def test_eval_does_not_count_or_balance_labels():
    trainer = object.__new__(AxolotlTrainer)
    trainer._train_batch_size = 2
    trainer.args = SimpleNamespace(
        sample_packing=True,
        eval_sample_packing=True,
        multipack_real_batches=False,
        per_device_train_batch_size=2,
        max_seq_length=8,
        balance_labels=True,
        seed=42,
        data_seed=7,
        world_size=2,
        gradient_accumulation_steps=4,
        sample_packing_efficiency=1,
        sample_packing_group_size=100,
        sample_packing_bin_size=8,
        sample_packing_sequentially=False,
        dataset_num_proc=1,
        sample_packing_mp_start_method="fork",
    )
    trainer.data_collator = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left")
    )
    data = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 32})
    with patch(
        "axolotl.core.trainers.base.get_dataset_label_counts",
        side_effect=AssertionError("eval labels scanned"),
    ):
        sampler = trainer._get_eval_sampler(data)
        assert sampler.label_counts is None
        assert sampler.dp_count == 1
        assert sampler.label_metrics is None
        assert sum(len(row) for batch in sampler for row in batch) == len(data)


def test_streaming_chunk_seed_is_content_based_and_replayable():
    import axolotl.utils.data.streaming as streaming
    from axolotl.utils.data.streaming import encode_packed_streaming

    seen = []

    def sampler(**kwargs):
        seen.append(kwargs["seed"])
        return []

    collator = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    def wrapper(dataset):
        return (dataset,)

    chunks = [
        dict(
            input_ids=[[i, i + 1] for i in range(start, start + 16)],
            attention_mask=[[1, 1]] * 16,
        )
        for start in (2, 20)
    ]
    with patch.object(streaming, "MultipackBatchSampler", side_effect=sampler):
        for seed, index in [(7, 0), (7, 1), (7, 0), (8, 0)]:
            encode_packed_streaming(
                collator,
                wrapper,
                chunks[index],
                bin_size=4,
                max_seq_length=8,
                batch_size=2,
                balance_labels=True,
                seed=seed,
            )
    assert seen[0] == seen[2]
    assert len(set(seen)) == 3
