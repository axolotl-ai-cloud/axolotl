"""Nemotron time weighting and logical reduction contracts."""

from __future__ import annotations

import torch

from axolotl.integrations.diffusion.lm.weighting import (
    reduce_objective,
    time_weights,
)
from axolotl.model_support.diffusion import ObjectiveReduction, TimeWeighting


def test_nemotron_inverse_time_weights_are_finite_at_the_floor():
    times = torch.tensor([0.001, 0.2, 0.75])
    torch.testing.assert_close(time_weights(times, TimeWeighting.INV_T), 1 / times)
    torch.testing.assert_close(time_weights(times, TimeWeighting.LINEAR), 1 - times)
    torch.testing.assert_close(
        time_weights(times, TimeWeighting.NONE), torch.ones_like(times)
    )


def test_reductions_preserve_unequal_logical_examples_and_empty_support():
    loss = torch.tensor([1.0, 3.0, 9.0], requires_grad=True)
    support = torch.tensor([True, True, True])
    ids = torch.tensor([0, 0, 1])
    token_mean = reduce_objective(loss, support, ObjectiveReduction.MASKED_TOKEN_MEAN)
    example_mean = reduce_objective(
        loss, support, ObjectiveReduction.EXAMPLE_MEAN, logical_ids=ids, logical_count=2
    )
    assert torch.equal(token_mean, torch.tensor(13 / 3))
    assert torch.equal(example_mean, torch.tensor(((1 + 3) / 2 + 9) / 2))
    with_empty = reduce_objective(
        loss,
        support,
        ObjectiveReduction.EXAMPLE_MEAN,
        logical_ids=torch.tensor([0, 0, -1]),
        logical_count=3,
    )
    assert torch.equal(with_empty, torch.tensor(((1 + 3) / 2) / 3))
    empty = reduce_objective(
        loss, torch.zeros_like(support), ObjectiveReduction.MASKED_TOKEN_MEAN
    )
    empty.backward()
    assert empty.item() == 0.0
    assert torch.equal(loss.grad, torch.zeros_like(loss))
