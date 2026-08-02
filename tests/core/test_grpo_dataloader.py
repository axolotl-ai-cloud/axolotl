from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from axolotl.core.trainers.grpo.trainer import AxolotlGRPOSequenceParallelTrainer


def _make_trainer(context_parallel_size, process_index, is_eval=False):
    trainer = AxolotlGRPOSequenceParallelTrainer.__new__(
        AxolotlGRPOSequenceParallelTrainer
    )
    trainer._train_batch_size = 2
    trainer.args = SimpleNamespace(
        eval_batch_size=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=None,
        sample_packing=False,
        pretraining=False,
        eval_sample_packing=None,
        dataloader_drop_last=False,
        process_index=process_index,
        context_parallel_size=context_parallel_size,
    )
    trainer.data_collator = lambda batch: {
        "input_ids": torch.tensor([[1, 2, 3] for _ in batch])
    }
    trainer.accelerator = SimpleNamespace(
        even_batches=True,
        prepare_data_loader=MagicMock(side_effect=lambda dl: dl),
    )
    return trainer


def test_cp_dataloader_not_prepared():
    trainer = _make_trainer(context_parallel_size=2, process_index=0)
    dataset = Dataset.from_dict({"a": [1, 2, 3, 4]})

    dataloader = trainer._prepare_dataloader(dataset, None, is_eval=False)

    assert isinstance(dataloader, DataLoader)
    trainer.accelerator.prepare_data_loader.assert_not_called()


def test_cp1_dataloader_prepared():
    trainer = _make_trainer(context_parallel_size=1, process_index=0)
    dataset = Dataset.from_dict({"a": [1, 2, 3, 4]})

    dataloader = trainer._prepare_dataloader(dataset, None, is_eval=False)

    trainer.accelerator.prepare_data_loader.assert_called_once_with(dataloader)


def test_cp_worker_seed_rank_is_sp_group():
    trainer = _make_trainer(context_parallel_size=2, process_index=3)
    dataset = Dataset.from_dict({"a": [1, 2, 3, 4]})

    dataloader = trainer._prepare_dataloader(dataset, None, is_eval=False)

    assert dataloader.worker_init_fn.keywords["rank"] == 1


def test_cp1_worker_seed_rank_is_process_index():
    trainer = _make_trainer(context_parallel_size=1, process_index=3)
    dataset = Dataset.from_dict({"a": [1, 2, 3, 4]})

    dataloader = trainer._prepare_dataloader(dataset, None, is_eval=False)

    assert dataloader.worker_init_fn.keywords["rank"] == 3


def test_cp_same_group_identical_worker_seed():
    group_rank0 = _make_trainer(context_parallel_size=4, process_index=0)
    group_rank3 = _make_trainer(context_parallel_size=4, process_index=3)
    dataset = Dataset.from_dict({"a": [1, 2, 3, 4]})

    dl_rank0 = group_rank0._prepare_dataloader(dataset, None, is_eval=False)
    dl_rank3 = group_rank3._prepare_dataloader(dataset, None, is_eval=False)

    assert (
        dl_rank0.worker_init_fn.keywords["rank"]
        == dl_rank3.worker_init_fn.keywords["rank"]
        == 0
    )


def test_cp_eval_has_no_worker_init_fn():
    trainer = _make_trainer(context_parallel_size=2, process_index=3)
    dataset = Dataset.from_dict({"a": [1, 2, 3, 4]})

    dataloader = trainer._prepare_dataloader(dataset, None, is_eval=True)

    assert dataloader.worker_init_fn is None
