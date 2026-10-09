"""Padded-batch costs and collated loss targets for fixed-count label balancing."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import BatchSampler
from transformers import PreTrainedTokenizerFast

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.samplers import LabelBalancedRandomSampler


def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


@pytest.mark.parametrize("multiple", [1, 8, 64])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_padded_cost_coverage_and_tail(multiple, batch_size):
    rng = np.random.default_rng(19)
    lengths = rng.integers(2, 129, 193)
    counts = np.array([rng.integers(0, length) for length in lengths])
    sampler = LabelBalancedRandomSampler(
        lengths,
        counts,
        batch_size,
        seed=42,
        padding_multiple=multiple,
        batches_per_optimizer_step=4,
    )
    before = torch.randperm(
        len(lengths), generator=torch.Generator().manual_seed(42)
    ).tolist()
    after = list(sampler)
    assert Counter(before) == Counter(after)
    cutoff = len(lengths) // batch_size // 4 * 4 * batch_size
    assert after[cutoff:] == before[cutoff:]
    a, b = sampler.label_metrics["before"], sampler.label_metrics["after"]
    assert b["std_label_count"] <= a["std_label_count"] + 1e-9
    assert b["total_batch_cost"] <= a["total_batch_cost"]
    assert b["padding_tokens"] <= a["padding_tokens"]
    assert b["max_batch_cost"] <= a["max_batch_cost"]
    batches = list(BatchSampler(sampler, batch_size, False))
    costs = [
        len(batch)
        * ((max(lengths[i] for i in batch) + multiple - 1) // multiple * multiple)
        for batch in batches
    ]
    assert sum(costs) == b["total_batch_cost"]
    assert b["std_batch_cost"] == pytest.approx(np.std(costs))
    before_full = [
        sampler._batch_cost(before[i : i + batch_size])
        for i in range(0, cutoff, batch_size)
    ]
    assert np.var(costs[: len(before_full)]) <= np.var(before_full) + 1e-9
    assert list(sampler) == after


def test_padded_mode_rejects_label_balance_that_adds_padding():
    sampler = LabelBalancedRandomSampler([8, 8, 1, 1], [4, 4, 1, 1], 2)
    before = [[0, 1], [2, 3]]
    assert sampler._balance_padded_window([row[:] for row in before]) == before
    flattened = sampler._balance_flattened_window([row[:] for row in before])
    assert [sum(sampler.label_counts[row]) for row in flattened] == [5, 5]
    assert sum(sampler._batch_cost(row) for row in flattened) > sum(
        sampler._batch_cost(row) for row in before
    )


@pytest.mark.parametrize("multiple", [1, 16, 128])
def test_actual_padded_collator_matches_sampler_metrics(multiple):
    examples = [
        {
            "input_ids": [2] * n,
            "labels": [2] + [2 if j % 3 else -100 for j in range(n - 1)],
        }
        for n in range(3, 67)
    ]
    dataset = Dataset.from_list(examples)
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        sample_packing=False,
        pretraining=False,
        batch_flattening=False,
        balance_labels=True,
        data_seed=7,
        seed=42,
        per_device_train_batch_size=4,
        world_size=1,
        gradient_accumulation_steps=4,
    )
    trainer.state = SimpleNamespace(train_batch_size=4)
    trainer.data_collator = DataCollatorForSeq2Seq(
        tokenizer(), pad_to_multiple_of=multiple
    )
    sampler = trainer._get_train_sampler(dataset)
    assert sampler.length_mode == "padded"
    assert sampler.padding_multiple == multiple
    costs, labels = [], []
    for batch in BatchSampler(sampler, 4, False):
        output = trainer.data_collator([dataset[i] for i in batch])
        assert output["input_ids"].ndim == 2
        assert output["input_ids"].shape[0] == 4
        costs.append(output["input_ids"].numel())
        labels.append(int((output["labels"][:, 1:] != -100).sum()))
    metrics = sampler.label_metrics["after"]
    assert sum(costs) == metrics["total_batch_cost"]
    assert np.mean(labels) == metrics["mean_label_count"]
    assert np.std(labels) == pytest.approx(metrics["std_label_count"])
    assert len(set(sampler.label_counts.tolist())) > 1
    trainer.data_collator.tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right-padding"):
        trainer._get_train_sampler(dataset)


def test_padded_configuration_without_varlen_attention():
    from axolotl.utils.schemas.config import AxolotlInputConfig

    cfg = AxolotlInputConfig(
        base_model="test",
        learning_rate=1e-5,
        datasets=[{"path": "test", "type": "alpaca"}],
        sample_packing=False,
        batch_flattening=False,
        balance_labels=True,
        micro_batch_size=4,
        attn_implementation="sdpa",
    )
    assert cfg.balance_labels
