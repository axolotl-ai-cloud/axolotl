"""Tests for MuonClip callback checkpointing."""

from pathlib import Path

import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from axolotl.muonclip import MuonClipController
from axolotl.utils.callbacks.muonclip import MuonClipCallback
from axolotl.utils.schemas.muon import MuonClipConfig

from .test_muonclip_controller import _TinyModel


def test_muonclip_callback_saves_and_restores(tmp_path):
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True, momentum=0.9)
    controller = MuonClipController(model, cfg, learning_rate=0.05)
    callback = MuonClipCallback(controller)

    param = model.linear.weight
    param.grad = torch.ones_like(param)
    controller.post_optimizer_step()

    args = TrainingArguments(output_dir=str(tmp_path), learning_rate=1e-4, per_device_train_batch_size=1)
    state = TrainerState()
    state.global_step = 2
    state.process_index = 0

    callback.on_save(args, state, TrainerControl())
    checkpoint_dir = Path(tmp_path) / "checkpoint-2"
    state_file = checkpoint_dir / "muonclip_state_rank0.pt"
    assert state_file.exists()

    before = controller.state_store.get_or_create(param).momentum.clone()
    controller.state_store.get_or_create(param).momentum.zero_()
    callback.load_state_from_checkpoint(checkpoint_dir)
    after = controller.state_store.get_or_create(param).momentum

    assert torch.allclose(before, after)
