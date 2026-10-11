# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Sampler replay validation must not prevent saving model checkpoints."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Trainer

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.samplers import LabelBalancedRandomSampler


def make_trainer(tmp_path):
    sampler = LabelBalancedRandomSampler(
        [4] * 128, [3] * 128, 2, batches_per_optimizer_step=4
    )
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        balance_labels=True,
        pretraining=False,
        ignore_data_skip=False,
        gradient_accumulation_steps=4,
        should_save=True,
        include_tkps=False,
    )
    trainer.state = SimpleNamespace(global_step=2, epoch=0.5)
    trainer.train_dataset = SimpleNamespace(_fingerprint="new")
    trainer._balanced_sampler_state = dict(
        version=1,
        settings=[trainer._balanced_sampler_settings(sampler)],
        dataset_fingerprint="old",
        steps_in_epoch=16,
        epoch=0,
    )
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer._save_gathered_lora_adapter = lambda *args: False
    trainer.is_deepspeed_enabled = False
    return trainer, SimpleNamespace(sampler=sampler)


def resume(trainer, loader, checkpoint):
    return trainer._run_epoch(
        None,
        0,
        loader,
        steps_in_epoch=16,
        epochs_trained=0,
        resume_from_checkpoint=str(checkpoint),
    )


@pytest.mark.parametrize("epoch", [0.3, None, float("nan"), float("inf")])
def test_invalid_replay_position_still_saves_checkpoint(tmp_path, monkeypatch, epoch):
    trainer, loader = make_trainer(tmp_path)
    trainer.state.epoch = epoch
    save = Mock(return_value="saved")
    monkeypatch.setattr(Trainer, "_save_checkpoint", save)
    assert trainer._save_checkpoint(torch.nn.Linear(2, 2), None) == "saved"
    save.assert_called_once()
    checkpoint = tmp_path / "checkpoint-2"
    metadata = json.loads((checkpoint / "balanced_sampler.json").read_text())
    assert metadata["replay_error"]
    trainer.state.epoch = 0.5
    run = Mock()
    monkeypatch.setattr(Trainer, "_run_epoch", run)
    with pytest.raises(ValueError, match="ignore_data_skip: true"):
        resume(trainer, loader, checkpoint)
    run.assert_not_called()
    trainer.args.ignore_data_skip = True
    resume(trainer, loader, checkpoint)
    run.assert_called_once()


@pytest.mark.parametrize("change", ["fingerprint", "metadata", "ranks"])
def test_resume_validates_schedule_but_only_warns_for_fingerprint(
    tmp_path, monkeypatch, change
):
    trainer, loader = make_trainer(tmp_path)
    saved = dict(trainer._balanced_sampler_state, resume_epoch=0, consumed_batches=8)
    if change == "metadata":
        saved["settings"][0]["metadata_sha256"] = "different"
    elif change == "ranks":
        saved["settings"][0]["dp_count"] = 2
    (tmp_path / "balanced_sampler.json").write_text(json.dumps(saved))
    run = Mock()
    warning = Mock()
    monkeypatch.setattr(Trainer, "_run_epoch", run)
    monkeypatch.setattr("axolotl.core.trainers.base.LOG.warning", warning)
    if change == "fingerprint":
        resume(trainer, loader, tmp_path)
        assert run.call_args.kwargs["steps_trained_in_current_epoch"] == 8
        warning.assert_called_once()
    else:
        with pytest.raises(ValueError, match="ignore_data_skip: true"):
            resume(trainer, loader, tmp_path)
        run.assert_not_called()
        trainer.args.ignore_data_skip = True
        resume(trainer, loader, tmp_path)
        run.assert_called_once()
