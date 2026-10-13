"""Tests for the deferred tokens-per-second accounting."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from axolotl.core.trainers.base import AxolotlTrainer


def _trainer(restored=None) -> AxolotlTrainer:
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(include_tkps=True)
    trainer.state = SimpleNamespace()
    if restored is not None:
        trainer.state.tokens = {k: torch.tensor(v) for k, v in restored.items()}
    trainer._tkps_trainable_acc = None
    trainer._tkps_total_acc = 0
    trainer._tkps_restored = (0, 0)
    trainer._tkps_state = None
    return trainer


def _batch(labels):
    labels = torch.tensor(labels)
    return {"input_ids": torch.zeros_like(labels), "labels": labels}


def test_accumulates_on_device_without_touching_state():
    trainer = _trainer()

    trainer._accumulate_token_counts(_batch([[1, 2, -100], [-100, -100, 3]]))
    trainer._accumulate_token_counts(_batch([[4, 5, 6]]))

    assert not hasattr(trainer.state, "tokens")
    assert trainer._tkps_trainable_acc.dtype == torch.int64
    assert trainer._tkps_trainable_acc.item() == 6
    assert trainer._tkps_total_acc == 9


def test_sync_materializes_state_tokens():
    trainer = _trainer()
    trainer._accumulate_token_counts(_batch([[1, -100, 2]]))

    trainer._sync_token_counts()

    assert trainer.state.tokens["trainable"].item() == 2
    assert trainer.state.tokens["total"].item() == 3


def test_sync_adds_restored_counts_once():
    trainer = _trainer(restored={"trainable": 100, "total": 250})
    trainer._accumulate_token_counts(_batch([[1, 2, 3]]))
    trainer._sync_token_counts()
    trainer._accumulate_token_counts(_batch([[4, -100]]))
    trainer._sync_token_counts()

    assert trainer.state.tokens["trainable"].item() == 104
    assert trainer.state.tokens["total"].item() == 255


def test_sync_is_noop_before_first_batch():
    trainer = _trainer()

    trainer._sync_token_counts()

    assert not hasattr(trainer.state, "tokens")


def test_counts_input_ids_when_no_labels():
    trainer = _trainer()

    trainer._accumulate_token_counts({"input_ids": torch.tensor([[7, 8, 9, 10]])})
    trainer._sync_token_counts()

    assert trainer.state.tokens["trainable"].item() == 4
    assert trainer.state.tokens["total"].item() == 4


def test_counts_keep_integer_precision():
    trainer = _trainer(restored={"trainable": 2**24, "total": 2**24})

    trainer._accumulate_token_counts(_batch([[1]]))
    trainer._sync_token_counts()

    assert trainer.state.tokens["trainable"].item() == 2**24 + 1


def test_sync_reduces_across_ranks_when_distributed():
    trainer = _trainer()
    trainer._accumulate_token_counts(_batch([[1, 2, -100]]))

    def fake_all_reduce(tensor, op=None):
        tensor.mul_(2)  # pretend a second rank contributed the same counts

    with (
        patch("axolotl.core.trainers.base.is_distributed", return_value=True),
        patch("torch.distributed.all_reduce", side_effect=fake_all_reduce) as reduce,
    ):
        trainer._sync_token_counts()

    reduce.assert_called_once()
    assert trainer.state.tokens["trainable"].item() == 4
    assert trainer.state.tokens["total"].item() == 6
