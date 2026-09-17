from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import GradientCheckpointingLayer

from axolotl.monkeypatch.checkpoint_activation_offload import (
    CheckpointHiddenStatesOffload,
)


class TinyCheckpointLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states):
        hidden_states = hidden_states.sin()
        return hidden_states * hidden_states


def test_checkpoint_offload_marks_non_reentrant_checkpoint_input():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = TinyCheckpointLayer()
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    layer.to(device)
    layer.train()
    hidden_states = torch.randn(4, 8, device=device, requires_grad=True)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_size=0)
    with offload:
        loss = layer(hidden_states).sum()
        loss.backward()

    assert hidden_states.grad is not None
    assert offload.stats.marked_tensors == 1
    assert offload.stats.saved_tensors_seen >= offload.stats.marked_tensors
    if hidden_states.device.type == "cuda":
        assert offload.stats.offloaded_tensors == 1
        assert offload.stats.restored_tensors == 1
    else:
        assert offload.stats.skipped_marked_tensors == 1


def test_checkpoint_offload_ignores_unmarked_saved_tensors():
    hidden_states = torch.randn(4, 8, requires_grad=True)
    linear = nn.Linear(8, 8)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_size=0)
    with offload:
        loss = linear(hidden_states).square().sum()
        loss.backward()

    assert hidden_states.grad is not None
    assert offload.stats.saved_tensors_seen > 0
    assert offload.stats.marked_tensors == 0
    assert offload.stats.offloaded_tensors == 0
