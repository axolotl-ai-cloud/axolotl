"""A strided EBFT batch too short for any rollout still yields a loss backward() accepts."""

import torch
from torch import nn

from axolotl.core.trainers.ebft.strided import AxolotlStridedEBFTTrainer


def test_too_short_sequence_returns_a_differentiable_zero_loss():
    trainer = object.__new__(AxolotlStridedEBFTTrainer)
    trainer.ebft_stride = 8
    trainer.ebft_context_length = 8
    trainer.ebft_generate_max_len = 8
    trainer.ebft_n_samples = 2
    trainer.ebft_min_completion_prefix = 8
    model = nn.Linear(2, 2)
    # the prompt fills the sequence, so there is no room for a completion block
    inputs = {
        "input_ids": torch.zeros(1, 16, dtype=torch.long),
        "prompt_length": torch.tensor([15]),
    }

    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    assert loss.item() == 0.0
    assert model.weight.grad is not None
    assert torch.count_nonzero(model.weight.grad) == 0
