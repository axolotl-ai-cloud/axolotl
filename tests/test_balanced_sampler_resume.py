# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Checkpoint skipping must restore the epoch before nested sharding wrappers."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.data_loader import (
    BatchSamplerShard,
    DataLoaderShard,
    skip_first_batches,
)
from torch.utils.data import BatchSampler, RandomSampler
from transformers import Trainer, TrainerCallback, TrainingArguments

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.training_args_base import AxolotlTrainingMixins
from axolotl.utils.samplers import LabelBalancedRandomSampler, MultipackBatchSampler


class IndexDataset:
    def __len__(self):
        return 259

    def __getitem__(self, index):
        return index


def identity(batch):
    return batch


def make_loader(kind, rank=0, replicas=1, variable_lengths=False):
    dataset = IndexDataset()
    lengths = np.full(len(dataset), 4)
    if variable_lengths:
        lengths = np.random.default_rng(7).integers(2, 9, len(dataset))
    counts = np.minimum(np.arange(len(dataset)) % 4, lengths)
    if kind == "packed":
        sampler = MultipackBatchSampler(
            RandomSampler(dataset),
            batch_size=1,
            batch_max_len=16,
            lengths=lengths,
            label_counts=counts,
            bin_size=4,
            num_processes=1,
            num_count_samples=1,
            seed=42,
            batches_per_optimizer_step=4 * replicas,
            dp_count=replicas,
        )
        len(sampler)
        batches = sampler
    else:
        sampler = LabelBalancedRandomSampler(
            lengths,
            counts,
            4,
            length_mode="flattened" if kind == "flattened" else "padded",
            seed=42,
            batches_per_optimizer_step=4 * replicas,
            dp_count=replicas,
        )
        batches = BatchSampler(sampler, 4, drop_last=True)
    shard = BatchSamplerShard(
        batches, num_processes=replicas, process_index=rank, even_batches=False
    )
    loader = DataLoaderShard(dataset, batch_sampler=shard, collate_fn=identity)
    return sampler, loader


@pytest.mark.parametrize("kind", ["packed", "flattened", "padded"])
@pytest.mark.parametrize("epoch", [0, 2])
@pytest.mark.parametrize("replicas", [1, 4])
@pytest.mark.parametrize("skip", [0, 4, 12])
def test_resumed_rank_batches_match_uninterrupted(
    monkeypatch, kind, epoch, replicas, skip
):
    # Mirror Trainer's wrapper order, retaining real Accelerate iteration and skipping.
    def run_epoch(self, model, epoch, train_dataloader, steps_trained_in_current_epoch):
        loader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
        loader.set_epoch(epoch)
        underlying = getattr(loader.batch_sampler, "batch_sampler", None)
        if hasattr(underlying, "set_epoch"):
            underlying.set_epoch(epoch)
        return list(loader)

    monkeypatch.setattr(Trainer, "_run_epoch", run_epoch)
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(balance_labels=True, pretraining=False)
    for rank in range(replicas):
        sampler, loader = make_loader(kind, rank, replicas)
        sampler.set_epoch(epoch)
        loader.set_epoch(epoch)
        expected = list(loader)
        torch.manual_seed(1234 + rank)
        restored, resumed_loader = make_loader(kind, rank, replicas)
        # Cached epoch-zero plans must not survive resume into a later epoch.
        list(restored)
        actual = trainer._run_epoch(
            model=None,
            epoch=epoch,
            train_dataloader=resumed_loader,
            steps_trained_in_current_epoch=skip,
        )
        assert restored.epoch == epoch
        assert actual == expected[skip:]
        assert actual
        assert restored.label_metrics == sampler.label_metrics


@dataclass
class ResumeTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    pass


class TraceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.trace = []
        self.config = SimpleNamespace()

    def forward(self, input_ids):
        self.trace.append(input_ids.tolist())
        loss = (self.weight - input_ids.float().mean() / 259).square()
        loss = loss * (0.9 + 0.2 * torch.rand((), device=loss.device))
        return {"loss": loss, "logits": self.weight.expand_as(input_ids)}


def tensor_collator(batch):
    if isinstance(batch[0], list):
        batch = [index for row in batch for index in row]
    return {"input_ids": torch.tensor(batch)}


@pytest.mark.parametrize("kind", ["packed", "flattened", "padded"])
@pytest.mark.parametrize("variable_lengths", [False, True])
@pytest.mark.parametrize("checkpoint_epoch", [1, 3])
def test_actual_checkpoint_resume(
    tmp_path, kind, variable_lengths, checkpoint_epoch, monkeypatch
):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _, initial_loader = make_loader(kind, variable_lengths=variable_lengths)
    save_step = checkpoint_epoch * ((len(initial_loader) + 3) // 4) + 1
    max_steps = save_step + 2

    class ResumeTrainer(AxolotlTrainer):
        def get_train_dataloader(self):
            _, loader = make_loader(kind, variable_lengths=variable_lengths)
            loader.base_dataloader.collate_fn = tensor_collator
            return loader

    def make_trainer(output_dir):
        return ResumeTrainer(
            model=TraceModel(),
            args=ResumeTrainingArguments(
                output_dir=str(output_dir),
                use_cpu=True,
                max_steps=max_steps,
                gradient_accumulation_steps=4,
                per_device_train_batch_size=4,
                balance_labels=True,
                save_steps=save_step,
                logging_strategy="no",
                report_to="none",
                disable_tqdm=True,
                dataloader_pin_memory=False,
            ),
            train_dataset=IndexDataset(),
        )

    original = make_trainer(tmp_path / "original")

    class SaveTrace(TrainerCallback):
        count = None

        def on_save(self, args, state, control, **kwargs):
            if state.global_step == save_step:
                self.count = len(original.model.trace)

    saved = SaveTrace()
    original.add_callback(saved)
    original.train()
    restored = make_trainer(tmp_path / "restored")
    restored.train(
        resume_from_checkpoint=str(tmp_path / "original" / f"checkpoint-{save_step}")
    )
    assert saved.count is not None
    assert restored.model.trace == original.model.trace[saved.count :]
    assert len(restored.model.trace) == 8
    assert restored.state.global_step == original.state.global_step == max_steps
    torch.testing.assert_close(
        restored.model.weight, original.model.weight, rtol=0, atol=0
    )


@pytest.mark.parametrize("kind", ["packed", "padded", "flattened"])
@pytest.mark.parametrize("window", [1, 8])
def test_checkpoint_resume_with_production_dataloader(
    tmp_path, kind, monkeypatch, window
):
    import json

    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import DataCollatorWithFlattening, PreTrainedTokenizerFast

    from axolotl.utils.collators import (
        BatchSamplerDataCollatorForSeq2Seq,
        DataCollatorForSeq2Seq,
    )

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [i + 2] * (3 + i % 6),
                "labels": [i + 2] * (3 + i % 6),
                "attention_mask": [1] * (3 + i % 6),
            }
            for i in range(131)
        ]
    )

    class Model(TraceModel):
        def forward(
            self, input_ids, labels=None, attention_mask=None, position_ids=None
        ):
            return super().forward(input_ids)

    def make(output, data_seed=7, balance_window=window):
        if kind == "packed":
            collator = BatchSamplerDataCollatorForSeq2Seq(tok, pad_to_multiple_of=8)
        elif kind == "flattened":
            collator = DataCollatorWithFlattening()
        else:
            collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8)
        return AxolotlTrainer(
            model=Model(),
            train_dataset=dataset,
            data_collator=collator,
            args=ResumeTrainingArguments(
                output_dir=str(output),
                use_cpu=True,
                max_steps=4,
                gradient_accumulation_steps=4,
                per_device_train_batch_size=4,
                balance_labels=True,
                sample_packing=kind == "packed",
                batch_flattening=kind == "flattened",
                max_seq_length=8,
                sample_packing_bin_size=8,
                sample_packing_group_size=100,
                dataset_num_proc=1,
                data_seed=data_seed,
                label_balance_window_optim_steps=balance_window,
                save_steps=2,
                logging_strategy="no",
                report_to="none",
                disable_tqdm=True,
                dataloader_pin_memory=False,
            ),
        )

    if kind == "packed":
        from axolotl.monkeypatch.data.batch_dataset_fetcher import _MapDatasetFetcher

        monkeypatch.setattr(
            torch.utils.data._utils.fetch, "_MapDatasetFetcher", _MapDatasetFetcher
        )
    original = make(tmp_path / "original")
    original.train()
    checkpoint = tmp_path / "original" / "checkpoint-2"
    metadata = json.loads((checkpoint / "balanced_sampler.json").read_text())
    assert metadata["consumed_batches"] == 8
    restored = make(tmp_path / "restored")
    restored.train(resume_from_checkpoint=str(checkpoint))
    assert restored.model.trace == original.model.trace[8:]
    torch.testing.assert_close(
        restored.model.weight, original.model.weight, rtol=0, atol=0
    )
    incompatible = make(tmp_path / "incompatible", data_seed=8)
    with pytest.raises(ValueError, match="resume changed settings"):
        incompatible.train(resume_from_checkpoint=str(checkpoint))

    changed_window = make(
        tmp_path / "changed-window", balance_window=8 if window == 1 else 1
    )
    with pytest.raises(ValueError, match="resume changed settings"):
        changed_window.train(resume_from_checkpoint=str(checkpoint))
