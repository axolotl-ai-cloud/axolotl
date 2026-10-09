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
from axolotl.utils.samplers import FlatteningAwareRandomSampler, MultipackBatchSampler


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
        )
        len(sampler)
        batches = sampler
    else:
        sampler = FlatteningAwareRandomSampler(
            lengths, counts, 4, seed=42, batches_per_optimizer_step=4 * replicas
        )
        batches = BatchSampler(sampler, 4, drop_last=True)
    shard = BatchSamplerShard(
        batches, num_processes=replicas, process_index=rank, even_batches=False
    )
    loader = DataLoaderShard(dataset, batch_sampler=shard, collate_fn=identity)
    return sampler, loader


@pytest.mark.parametrize("kind", ["packed", "flattened"])
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
    trainer.args = SimpleNamespace(balance_packed_labels=True, pretraining=False)
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


@pytest.mark.parametrize("kind", ["packed", "flattened"])
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
                balance_packed_labels=True,
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
