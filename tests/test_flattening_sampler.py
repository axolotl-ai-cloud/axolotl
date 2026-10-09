"""Fixed-count balancing, collation and distributed sharding regressions."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from datasets import Dataset
from torch.utils.data import BatchSampler
from transformers import DataCollatorWithFlattening

from axolotl.utils.samplers import LabelBalancedRandomSampler


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
@pytest.mark.parametrize("size", [0, 3, 67, 521])
def test_cardinality_coverage_objective_and_tail(batch_size, size):
    rng = np.random.default_rng(42)
    lengths = rng.integers(2, 129, size)
    counts = np.array([rng.integers(0, n) for n in lengths], dtype=np.int64)
    sampler = LabelBalancedRandomSampler(
        lengths,
        counts,
        batch_size,
        length_mode="flattened",
        seed=42,
        batches_per_optimizer_step=4,
    )
    before = torch.randperm(size, generator=torch.Generator().manual_seed(42)).tolist()
    after = list(sampler)
    assert Counter(after) == Counter(before)
    cut = (size // batch_size // 4) * 4 * batch_size
    assert after[cut:] == before[cut:]
    batches = list(BatchSampler(sampler, batch_size, drop_last=False))
    assert all(len(b) == batch_size for b in batches[:-1])
    a, b = sampler.label_metrics["before"], sampler.label_metrics["after"]
    assert b["std_label_count"] <= a["std_label_count"] + 1e-10
    assert b["max_unpadded_length"] <= a["max_unpadded_length"]
    assert b["std_unpadded_length"] <= a["std_unpadded_length"] + 1e-10
    assert b["mean_unpadded_length"] == a["mean_unpadded_length"]
    assert b["total_label_count"] == a["total_label_count"]
    assert list(sampler) == after


def test_collated_labels_and_strong_balance():
    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1] + ([2, 3, 4] if i < 32 else [-100, -100, -100]),
        }
        for i in range(64)
    ]
    sampler = LabelBalancedRandomSampler(
        [4] * 64, [3] * 32 + [0] * 32, 4, length_mode="flattened", seed=42
    )
    collator = DataCollatorWithFlattening()
    labels = []
    for batch in BatchSampler(sampler, 4, drop_last=True):
        result = collator([examples[i] for i in batch])
        assert result["input_ids"].shape == (1, 16)
        assert result["position_ids"].tolist() == [[0, 1, 2, 3] * 4]
        labels.append(int((result["labels"][:, 1:] != -100).sum()))
    assert labels == [6] * 16
    assert sampler.label_metrics["after"]["mean_label_count"] == 6
    assert sampler.label_metrics["after"]["std_label_count"] == 0
    assert sampler.label_metrics["before"]["std_label_count"] > 0


def test_rank_rng_epoch_and_sharding():
    plans, shards = [], []
    for rank in range(4):
        with torch.random.fork_rng():
            torch.manual_seed(rank + 100)
            sampler = LabelBalancedRandomSampler(
                [4] * 132,
                [0, 1, 2, 3] * 33,
                4,
                length_mode="flattened",
                seed=42,
                batches_per_optimizer_step=4,
            )
            state = torch.get_rng_state().clone()
            first = list(sampler)
            assert torch.equal(state, torch.get_rng_state())
            sampler.set_epoch(1)
            second = list(sampler)
            assert second != first
            sampler.set_epoch(0)
            assert list(sampler) == first
            plans.append(first)
            batches = BatchSampler(sampler, 4, drop_last=True)
            shards.append(
                list(
                    BatchSamplerShard(
                        batches, num_processes=4, process_index=rank, even_batches=False
                    )
                )
            )
    assert all(p == plans[0] for p in plans)
    assert {len(s) for s in shards} == {8}
    retained = [i for shard in shards for batch in shard for i in batch]
    baseline = torch.randperm(132, generator=torch.Generator().manual_seed(42)).tolist()
    assert Counter(retained) == Counter(baseline[:128])


@pytest.mark.parametrize("provided_labels", [True, False])
def test_trainer_selects_fixed_count_sampler(provided_labels):
    from axolotl.core.trainers.base import AxolotlTrainer

    data = {"input_ids": [[1, 2, 3, 4]] * 16}
    if provided_labels:
        data["labels"] = [[1, -100, 3, 4]] * 16
    dataset = Dataset.from_dict(data)
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        sample_packing=False,
        pretraining=False,
        batch_flattening=True,
        balance_labels=True,
        data_seed=7,
        seed=42,
        per_device_train_batch_size=8,
        world_size=4,
        gradient_accumulation_steps=4,
    )
    trainer.state = SimpleNamespace(train_batch_size=4)
    trainer.data_collator = DataCollatorWithFlattening()
    sampler = trainer._get_train_sampler(dataset)
    assert isinstance(sampler, LabelBalancedRandomSampler)
    assert sampler.batch_size == 4
    assert sampler.seed == 7
    assert sampler.batches_per_optimizer_step == 16
    assert sampler.label_counts.tolist() == [2 if provided_labels else 3] * 16
    assert len(list(sampler)) == 16
    trainer.data_collator = DataCollatorWithFlattening(separator_id=0)
    with pytest.raises(ValueError, match="separator_id"):
        trainer._get_train_sampler(dataset)


def test_flattening_config():
    from axolotl.utils.schemas.config import AxolotlInputConfig

    cfg = dict(
        base_model="test",
        learning_rate=1e-5,
        datasets=[{"path": "test", "type": "alpaca"}],
        sample_packing=False,
        batch_flattening=True,
        attn_implementation="varlen",
        micro_batch_size=4,
        balance_labels=True,
    )
    assert AxolotlInputConfig(**cfg).balance_labels
    for change in [
        {"streaming": True, "max_steps": 10},
        {"group_by_length": True},
        {"curriculum_sampling": True},
        {"accelerator_config": {"split_batches": True}},
    ]:
        with pytest.raises(ValueError, match="balance_labels"):
            AxolotlInputConfig(**(cfg | change))


def test_iterable_flattening_balance_rejected_at_dataloader():
    from axolotl.core.trainers.base import AxolotlTrainer

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([])

    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        balance_labels=True, batch_flattening=True, sample_packing=False
    )
    with pytest.raises(ValueError, match="map-style"):
        trainer._get_dataloader(Stream(), "training", 4, is_training=True)
