"""Tests for weighted-loss helpers in axolotl.utils.trainer"""

import math

import pytest
import torch

from axolotl.utils.trainer import (
    create_weighted_mask,
    trainer_weighted_loss,
    weighted_cross_entropy,
)


def reference_weighted_mask(labels: torch.Tensor) -> torch.Tensor:
    """Naive reference: each contiguous run of unmasked (!= -100) labels is a
    group, and every token in a group gets weight 1/len(group)."""
    labels_2d = labels.unsqueeze(0) if labels.dim() == 1 else labels
    weights = torch.zeros_like(labels_2d, dtype=torch.float32)
    for i in range(labels_2d.shape[0]):
        group_positions = []
        current: list[int] | None = None
        for j, val in enumerate(labels_2d[i].tolist()):
            if val != -100:
                if current is None:
                    current = []
                    group_positions.append(current)
                current.append(j)
            else:
                current = None
        for group in group_positions:
            for j in group:
                weights[i, j] = 1.0 / len(group)
    return weights.squeeze()


def test_create_weighted_mask_1d():
    labels = torch.tensor([-100, 5, 6, -100, -100, 7, 8, 9])
    expected = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3])

    weights = create_weighted_mask(labels)

    assert weights.shape == labels.shape
    torch.testing.assert_close(weights, expected)


def test_create_weighted_mask_row_starting_unmasked():
    labels = torch.tensor([1, 2, -100, 3])
    expected = torch.tensor([0.5, 0.5, 0.0, 1.0])

    torch.testing.assert_close(create_weighted_mask(labels), expected)


def test_create_weighted_mask_batch():
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, -100, 4],
            [5, -100, -100, 6, 7, -100],
        ]
    )
    expected = torch.tensor(
        [
            [0.0, 1 / 3, 1 / 3, 1 / 3, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        ]
    )

    weights = create_weighted_mask(labels)

    assert weights.shape == labels.shape
    torch.testing.assert_close(weights, expected)


def test_create_weighted_mask_all_masked_row():
    labels = torch.tensor(
        [
            [-100, -100, -100, -100],
            [-100, 1, 2, -100],
        ]
    )
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
        ]
    )

    torch.testing.assert_close(create_weighted_mask(labels), expected)


def test_create_weighted_mask_all_unmasked_row():
    labels = torch.tensor([[1, 2, 3, 4]])
    # batch size 1 output is squeezed to 1D, matching the historical behavior
    expected = torch.tensor([0.25, 0.25, 0.25, 0.25])

    weights = create_weighted_mask(labels)

    assert weights.shape == expected.shape
    torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_create_weighted_mask_matches_reference_random(seed):
    torch.manual_seed(seed)
    labels = torch.randint(0, 100, (4, 64))
    labels[torch.rand(labels.shape) < 0.4] = -100

    torch.testing.assert_close(
        create_weighted_mask(labels), reference_weighted_mask(labels)
    )


def test_weighted_cross_entropy_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))
    labels[:, :3] = -100
    weights = create_weighted_mask(labels)

    loss = weighted_cross_entropy(logits, labels, weights)

    manual = (
        torch.nn.functional.cross_entropy(
            logits.view(-1, 32), labels.view(-1), reduction="none"
        )
        * weights.view(-1)
    ).sum()
    torch.testing.assert_close(loss, manual)


def test_weighted_cross_entropy_uniform_logits():
    # uniform logits: CE is ln(V) per token, and each group's weights sum to 1,
    # so the weighted sum is ln(V) * num_groups
    vocab_size = 64
    logits = torch.zeros(1, 6, vocab_size)
    labels = torch.tensor([[-100, 1, 2, -100, 3, 4]])  # 2 groups
    weights = create_weighted_mask(labels)

    loss = weighted_cross_entropy(logits, labels, weights)

    torch.testing.assert_close(loss, torch.tensor(2 * math.log(vocab_size)))


@pytest.mark.parametrize("output_as_dict", [True, False])
def test_trainer_weighted_loss_shifts_labels(output_as_dict):
    torch.manual_seed(0)
    logits = torch.randn(2, 9, 32)
    labels = torch.randint(0, 32, (2, 9))
    labels[:, :4] = -100
    model_output = {"logits": logits} if output_as_dict else (logits,)

    loss = trainer_weighted_loss(model_output, labels, shift_labels=True)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    weights = reference_weighted_mask(shift_labels)
    expected = (
        torch.nn.functional.cross_entropy(
            shift_logits.view(-1, 32), shift_labels.view(-1), reduction="none"
        )
        * weights.view(-1)
    ).sum()
    torch.testing.assert_close(loss, expected)


def test_trainer_weighted_loss_grad_flows():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 32, requires_grad=True)
    labels = torch.randint(0, 32, (2, 8))
    labels[:, :2] = -100

    loss = trainer_weighted_loss({"logits": logits}, labels, shift_labels=True)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
