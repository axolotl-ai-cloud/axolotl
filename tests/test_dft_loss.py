"""
test suite for dynamic fine-tuning loss
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.monkeypatch.loss.dft import dft_loss


@pytest.fixture
def dft_fixtures():
    torch.manual_seed(0)
    vocab_size, seq_len, batch_size = 64, 16, 2
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :4] = -100
    return logits, labels, vocab_size


def _reference(logits, labels):
    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[..., 1:].reshape(-1)
    mask = shift_labels != -100
    ce = nn.functional.cross_entropy(
        shift_logits[mask], shift_labels[mask], reduction="none"
    )
    probs = torch.softmax(shift_logits[mask], dim=-1).gather(
        -1, shift_labels[mask].unsqueeze(-1)
    )
    return (probs.squeeze(-1).detach() * ce), mask.sum()


def test_dft_matches_probability_weighted_ce(dft_fixtures):
    logits, labels, _ = dft_fixtures
    weighted_ce, num_tokens = _reference(logits, labels)

    loss = dft_loss(SimpleNamespace(logits=logits), labels)
    assert torch.allclose(loss, weighted_ce.mean())

    loss = dft_loss(SimpleNamespace(logits=logits), labels, num_items_in_batch=7)
    assert torch.allclose(loss, weighted_ce.sum() / 7)
    assert num_tokens == (labels[..., 1:] != -100).sum()


def test_dft_gradient_is_probability_scaled_ce_gradient(dft_fixtures):
    logits, labels, _ = dft_fixtures

    dft_grad = torch.autograd.grad(
        dft_loss(SimpleNamespace(logits=logits), labels, num_items_in_batch=1),
        logits,
    )[0]
    weighted_ce, _ = _reference(logits, labels)
    ref_grad = torch.autograd.grad(weighted_ce.sum(), logits)[0]

    assert torch.allclose(dft_grad, ref_grad, atol=1e-6)


def test_dft_all_masked_is_zero():
    logits = torch.randn(1, 8, 32, requires_grad=True)
    labels = torch.full((1, 8), -100)

    loss = dft_loss(SimpleNamespace(logits=logits), labels)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(logits.grad == 0)
