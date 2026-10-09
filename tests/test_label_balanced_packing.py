"""Coverage and objective regressions for supervised-token balanced packing."""

from collections import Counter

import numpy as np
import pytest
from datasets import Dataset

from axolotl.utils.samplers import MultipackBatchSampler
from axolotl.utils.samplers.label_balance import balance_labels
from axolotl.utils.samplers.utils import get_dataset_label_counts


def labels_per_batch(batches, counts, starts):
    return [
        sum(sum(int(counts[i]) for i in bin_) - int(starts[bin_[0]]) for bin_ in batch)
        for batch in batches
    ]


def coverage(batches):
    return Counter(i for batch in batches for bin_ in batch for i in bin_)


def make_sampler(lengths, counts=None, starts=None, batch_size=1, **kwargs):
    return MultipackBatchSampler(
        sampler=list(range(len(lengths))),
        lengths=np.array(lengths),
        label_counts=None if counts is None else np.array(counts),
        label_start_counts=None if starts is None else np.array(starts),
        batch_size=batch_size,
        batch_max_len=8,
        bin_size=8,
        num_processes=1,
        **kwargs,
    )


def test_full_bins_balance_without_padding_cost():
    sampler = make_sampler([4] * 4, [3, 3, 1, 1])
    batches = sampler.generate_batches(set_stats=True)
    assert labels_per_batch(batches, [3, 3, 1, 1], [0] * 4) == [4, 4]
    assert coverage(batches) == Counter(range(4))
    assert sampler.efficiency() == 1.0


def test_whole_bins_balance_real_batches():
    sampler = make_sampler([8] * 4, [7, 7, 1, 1], batch_size=2)
    batches = sampler.generate_batches()
    assert labels_per_batch(batches, [7, 7, 1, 1], [0] * 4) == [8, 8]
    assert all(len(batch) == 2 for batch in batches)


def test_repacking_escapes_one_for_one_swap_constraint():
    lengths = np.array([2, 1, 3, 5, 1, 1])
    counts = np.array([1, 1, 3, 0, 1, 1])
    batches = [[[0, 1, 2]], [[3, 4, 5]]]
    balanced = balance_labels(batches, lengths, counts, np.zeros(6, dtype=int), 8, 8, 0)
    assert coverage(balanced) == coverage(batches)
    assert sorted(labels_per_batch(balanced, counts, [0] * 6)) == [3, 4]
    assert sorted(len(batch[0]) for batch in balanced) == [2, 4]
    constrained = balance_labels(
        batches, lengths, counts, np.zeros(6, dtype=int), 8, 3, 0
    )
    assert coverage(constrained) == coverage(batches)
    assert all(len(bin_) <= 3 for batch in constrained for bin_ in batch)


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("padding_multiple", [None, 4])
def test_randomized_capacity_coverage_and_monotonic_objective(
    drop_last, batch_size, seed, padding_multiple
):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 9, size=401)
    counts = np.array([rng.integers(0, length + 1) for length in lengths])
    starts = ((counts > 0) & (rng.random(len(lengths)) < 0.5)).astype(int)
    original = make_sampler(lengths, batch_size=batch_size, drop_last=drop_last)
    balanced = make_sampler(
        lengths,
        counts,
        starts,
        batch_size=batch_size,
        drop_last=drop_last,
        padding_multiple=padding_multiple,
    )
    before = original.generate_batches(set_stats=True)
    after = balanced.generate_batches(set_stats=True)
    assert coverage(before) == coverage(after)
    assert len(before) == len(after)
    assert original.efficiency() == balanced.efficiency()
    assert all(sum(lengths[i] for i in bin_) <= 8 for batch in after for bin_ in batch)
    assert all(0 < len(bin_) <= 8 for batch in after for bin_ in batch)
    before_labels = labels_per_batch(before, counts, starts)
    after_labels = labels_per_batch(after, counts, starts)
    assert sum(before_labels) == sum(after_labels)
    assert sum(x * x for x in after_labels) <= sum(x * x for x in before_labels)
    if padding_multiple is not None:

        def padded_slots(batches):
            return sum(
                (
                    (
                        max(sum(lengths[i] for i in bin_) for bin_ in batch)
                        + padding_multiple
                        - 1
                    )
                    // padding_multiple
                )
                * padding_multiple
                * len(batch)
                for batch in batches
            )

        assert padded_slots(after) <= padded_slots(before)
    if before and len(before[-1]) < batch_size:
        assert after[-1] == before[-1]


def test_counts_match_concatenated_causal_labels():
    dataset = Dataset.from_dict({"labels": [[1, -100, 2], [-100, 3], [4, 5], []]})
    counts, starts = get_dataset_label_counts(dataset)
    assert counts.tolist() == [2, 1, 2, 0]
    assert starts.tolist() == [1, 0, 1, 0]
    for bin_ in [[0, 1, 2], [1, 0, 2], [2, 1, 0]]:
        labels = np.concatenate([dataset[i]["labels"] for i in bin_])
        assert labels_per_batch([[bin_]], counts, starts) == [
            np.count_nonzero(labels[1:] != -100)
        ]
    _, unshifted = get_dataset_label_counts(dataset, shift_labels=False)
    assert not unshifted.any()
    shifted = dataset.rename_column("labels", "shift_labels")
    shifted_counts, shifted_starts = get_dataset_label_counts(shifted)
    np.testing.assert_array_equal(shifted_counts, counts)
    assert not shifted_starts.any()


def test_missing_labels_rejected():
    with pytest.raises(ValueError, match="tokenized labels"):
        get_dataset_label_counts(Dataset.from_dict({"input_ids": [[1, 2]]}))


def test_deterministic_epoch_and_cache():
    sampler = make_sampler([4] * 40, [3, 3, 1, 1] * 10, seed=42)
    first = sampler.generate_batches()
    assert sampler.generate_batches() is first
    sampler.set_epoch(0)
    assert sampler.generate_batches() == first
    sampler.set_epoch(1)
    second = sampler.generate_batches()
    assert second != first
    assert coverage(second) == coverage(first)


@pytest.mark.parametrize(
    "counts, starts",
    [
        ([-1, 2], None),
        ([5, 2], None),
        ([1], None),
        ([1.5, 2], None),
        ([1, 2], [2, 0]),
        ([0, 2], [1, 0]),
    ],
)
def test_invalid_metadata_rejected(counts, starts):
    with pytest.raises(ValueError, match="Invalid lengths or label counts"):
        make_sampler([4, 4], counts, starts)


def test_sequential_mode_rejected():
    with pytest.raises(ValueError, match="sequential"):
        make_sampler([4, 4], [1, 2], sequential=True)


def test_zero_labels_empty_input_and_indivisible_samples():
    for lengths, counts in [([], []), ([8, 8], [0, 0]), ([8, 8], [8, 0])]:
        # Explicit dtype is needed for the empty integer metadata.
        sampler = make_sampler(lengths, np.array(counts, dtype=int))
        batches = sampler.generate_batches()
        assert coverage(batches) == Counter(range(len(lengths)))


def test_nonidentity_sampler_indices():
    sampler = make_sampler([4] * 6, [3, 1, 2, 1, 3, 2])
    sampler.sampler = [4, 0, 3, 1]
    batches = sampler.generate_batches()
    assert coverage(batches) == Counter([4, 0, 3, 1])
    assert labels_per_batch(batches, sampler.label_counts, [0] * 6) == [4, 4]


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_packing_sequentially": True},
        {"curriculum_sampling": True},
        {"reward_model": True},
        {"process_reward_model": True},
        {"diffusion_lm": {}},
    ],
)
def test_incompatible_config_rejected(overrides):
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = dict(
        base_model="test",
        learning_rate=1e-5,
        datasets=[{"path": "test", "type": "alpaca"}],
        sample_packing=True,
        balance_labels=True,
    )
    config.update(overrides)
    with pytest.raises(ValueError, match="balance_labels"):
        AxolotlInputConfig(**config)


def test_config_accepts_opt_in_and_defaults_off():
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = dict(
        base_model="test",
        learning_rate=1e-5,
        datasets=[{"path": "test", "type": "alpaca"}],
        sample_packing=True,
    )
    assert not AxolotlInputConfig(**config).balance_labels
    assert AxolotlInputConfig(**config, balance_labels=True).balance_labels


@pytest.mark.parametrize("real_batches", [False, True])
@pytest.mark.parametrize("shift_labels", [False, True])
def test_trainer_passes_label_metadata(real_batches, shift_labels):
    from types import SimpleNamespace

    from axolotl.core.trainers.base import AxolotlTrainer

    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * 8, "labels": [[1, -100, 3, 4]] * 8}
    )
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        multipack_real_batches=real_batches,
        per_device_train_batch_size=2,
        max_seq_length=8,
        balance_labels=True,
        sample_packing_efficiency=1.0,
        sample_packing_group_size=100,
        sample_packing_bin_size=8,
        sample_packing_sequentially=False,
        dataset_num_proc=1,
        sample_packing_mp_start_method="fork",
        seed=42,
        gradient_accumulation_steps=4,
        world_size=4,
    )
    trainer.state = SimpleNamespace(train_batch_size=2)
    trainer._loss_shifts_labels = shift_labels
    sampler = trainer._create_multipack_sampler(list(range(8)), dataset)
    assert sampler.batches_per_optimizer_step == 16
    assert sampler.label_counts.tolist() == [3] * 8
    assert sampler.label_start_counts.tolist() == [int(shift_labels)] * 8
    assert sampler.batch_max_len == (8 if real_batches else 16)
    assert sampler.batch_size == (2 if real_batches else 1)
    assert coverage(sampler.generate_batches()) == Counter(range(8))
    trainer.data_collator = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left")
    )
    with pytest.raises(ValueError, match="right padding"):
        trainer._create_multipack_sampler(list(range(8)), dataset)
    trainer.args.balance_labels = False
    sampler = trainer._create_multipack_sampler(
        list(range(8)), dataset.remove_columns("labels")
    )
    assert sampler.label_counts is None


@pytest.mark.parametrize("pretraining", [False, True])
def test_streaming_config_accepts_label_balancing(pretraining):
    from axolotl.utils.schemas.config import AxolotlInputConfig

    config = dict(
        base_model="test",
        learning_rate=1e-5,
        sample_packing=True,
        balance_labels=True,
        max_steps=10,
    )
    if pretraining:
        config["pretraining_dataset"] = [{"path": "test"}]
        config["streaming"] = True
    else:
        config.update(streaming=True, datasets=[{"path": "test", "type": "alpaca"}])
    assert AxolotlInputConfig(**config).balance_labels


@pytest.mark.parametrize("pretraining", [False, True])
@pytest.mark.parametrize("explicit_labels", [False, True])
@pytest.mark.parametrize("multipack_attn", [False, True])
def test_streaming_balances_each_chunk(pretraining, explicit_labels, multipack_attn):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from axolotl.utils.data.streaming import wrap_streaming_dataset
    from axolotl.utils.dict import DictDefault

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
        padding_side="right",
    )
    examples = []
    for i in range(26):
        example = {"input_ids": [i + 2] * 4, "attention_mask": [1] * 4}
        if explicit_labels:
            count = 3 if i % 8 < 4 else 1
            example["labels"] = [-100] * (4 - count) + [i + 2] * count
        examples.append(example)

    def run(balance):
        calls = []

        def wrapper(dataset):
            calls.append(len(dataset))
            return (dataset,)

        cfg = DictDefault(
            dict(
                sample_packing=True,
                balance_labels=balance,
                pretraining_dataset="test" if pretraining else None,
                pretrain_multipack_attn=multipack_attn,
                sequence_len=4,
                micro_batch_size=2,
                streaming_multipack_buffer_size=8,
                sample_packing_bin_size=2,
                seed=42,
            )
        )
        with torch.random.fork_rng():
            torch.manual_seed(0)
            dataset = Dataset.from_list(examples).to_iterable_dataset()
            wrapped = wrap_streaming_dataset(dataset, tokenizer, cfg, wrapper)
            assert calls == []
            output = [
                {
                    key: value.tolist() if torch.is_tensor(value) else value
                    for key, value in row.items()
                }
                for row in wrapped
            ]
        assert calls == [8, 8, 8, 2]
        assert cfg.micro_batch_size == 1
        return output

    before, after = run(False), run(True)
    assert after == run(True)
    assert len(before) == len(after) == 13
    assert Counter(after[-1]["input_ids"]) == Counter([26] * 4 + [27] * 4)
    assert sum(label != -100 for label in after[-1]["labels"][1:]) == (
        6 if explicit_labels else 7
    )
    for chunk in range(3):
        old = before[chunk * 4 : (chunk + 1) * 4]
        new = after[chunk * 4 : (chunk + 1) * 4]
        old_tokens = Counter(token for row in old for token in row["input_ids"])
        new_tokens = Counter(token for row in new for token in row["input_ids"])
        assert old_tokens == new_tokens
        assert set(new_tokens) == set(range(chunk * 8 + 2, chunk * 8 + 10))
        before_counts = [
            sum(label != -100 for label in row["labels"][1:]) for row in old
        ]
        after_counts = [
            sum(label != -100 for label in row["labels"][1:]) for row in new
        ]
        assert sum(before_counts) == sum(after_counts)
        assert np.var(after_counts) <= np.var(before_counts)
        if explicit_labels:
            assert after_counts == [4] * 4
        else:
            assert after_counts == [7] * 4
        for row in new:
            assert len(row["input_ids"]) == len(row["labels"]) == 8
            if multipack_attn or not pretraining:
                assert row["position_ids"] == [0, 1, 2, 3] * 2
            else:
                assert row["attention_mask"] == [1] * 8


def test_balancing_does_not_spread_slack_into_extra_padding():
    lengths = np.array([6, 2, 4])
    counts = np.array([6, 2, 0])
    starts = np.zeros(3, dtype=int)
    batches = [[[0, 1]], [[2]]]
    balanced = balance_labels(
        batches, lengths, counts, starts, 8, 8, 0, padding_multiple=4
    )
    assert sorted(labels_per_batch(balanced, counts, starts)) == [0, 8]
    assert sorted(sum(lengths[i] for i in batch[0]) for batch in balanced) == [4, 8]
    unconstrained = balance_labels(batches, lengths, counts, starts, 8, 8, 0)
    assert sorted(sum(lengths[i] for i in batch[0]) for batch in unconstrained) == [
        6,
        6,
    ]


@pytest.mark.parametrize("drop_last", [False, True])
def test_label_metrics_match_generated_batches(drop_last):
    sampler = make_sampler(
        [4] * 9, [3, 3, 1, 1, 2, 2, 4, 1, 2], batch_size=2, drop_last=drop_last
    )
    batches = sampler.generate_batches()
    metrics = sampler.label_metrics
    assert metrics is not None
    counts = labels_per_batch(batches, sampler.label_counts, sampler.label_start_counts)
    assert metrics["after"]["mean_label_count"] == np.mean(counts)
    assert metrics["after"]["std_label_count"] == np.std(counts)
    assert metrics["after"]["total_label_count"] == sum(counts)
    assert metrics["after"]["mean_packed_length"] == np.mean(
        [sum(sampler.lengths[i] for i in bin_) for batch in batches for bin_ in batch]
    )
    assert (
        metrics["before"]["total_label_count"] == metrics["after"]["total_label_count"]
    )
    assert metrics["before"]["std_label_count"] >= metrics["after"]["std_label_count"]
    assert sampler.generate_batches(set_stats=True) is batches
    assert sampler.label_metrics is metrics
    assert sampler._label_metrics_logged
    sampler.set_epoch(1)
    assert sampler.label_metrics is None
    assert not sampler._label_metrics_logged


def test_empty_label_metrics():
    sampler = make_sampler([4], [3], batch_size=2)
    assert sampler.generate_batches(set_stats=True) == []
    assert all(value == 0 for value in sampler.label_metrics["after"].values())


def test_balanced_random_sampler_ignores_rank_local_rng():
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import RandomSampler

    plans = []
    shards = []
    for rank in range(4):
        torch.manual_seed(100 + rank)
        np.random.seed(200 + rank)
        sampler = MultipackBatchSampler(
            RandomSampler(range(128)),
            lengths=np.full(128, 4),
            label_counts=np.tile([1, 2, 3, 4], 32),
            batch_size=2,
            batch_max_len=8,
            bin_size=8,
            num_processes=1,
            seed=42,
        )
        state = torch.get_rng_state().clone()
        first = sampler.generate_batches()
        assert torch.equal(state, torch.get_rng_state())
        sampler.set_epoch(0)
        assert sampler.generate_batches() == first
        sampler.set_epoch(1)
        second = sampler.generate_batches()
        assert first != second
        plans.append((first, second))
        shards.append(
            list(
                BatchSamplerShard(
                    sampler, num_processes=4, process_index=rank, even_batches=False
                )
            )
        )
    assert all(plan == plans[0] for plan in plans)
    assert len({len(shard) for shard in shards}) == 1
    assert Counter(
        i for shard in shards for batch in shard for bin_ in batch for i in bin_
    ) == Counter(range(128))
