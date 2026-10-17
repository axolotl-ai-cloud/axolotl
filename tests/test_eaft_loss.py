"""Tests for the entropy-aware focal training loss."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from axolotl.monkeypatch.loss.eaft import eaft_loss

from tests.host_sync_utils import HostSyncRecorder


def _reference(logits, labels, num_items_in_batch=None, alpha=1.0, k=20):
    """The boolean-mask implementation eaft_loss replaced."""
    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    mask = shift_labels != -100
    with torch.no_grad():
        top_k_logits, _ = torch.topk(
            shift_logits[mask].float(), k=min(k, vocab_size), dim=-1
        )
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        entropy = -(top_k_probs * torch.log(top_k_probs + 1e-10)).sum(dim=-1)
        weights = torch.pow(entropy, alpha)
    per_token_loss = torch.nn.CrossEntropyLoss(reduction="none")(
        shift_logits[mask], shift_labels[mask]
    )
    weighted_loss = per_token_loss * weights
    if num_items_in_batch is not None:
        return weighted_loss.sum() / num_items_in_batch
    return weighted_loss.mean()


@pytest.fixture
def eaft_fixtures():
    torch.manual_seed(0)
    vocab_size, seq_len, batch_size = 64, 16, 2
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :5] = -100
    labels[1, 9:11] = -100
    return logits, labels


@pytest.mark.parametrize("num_items_in_batch", [None, 7])
@pytest.mark.parametrize("alpha, k", [(1.0, 20), (0.5, 8), (2.0, 64)])
def test_eaft_matches_masked_reference(eaft_fixtures, num_items_in_batch, alpha, k):
    logits, labels = eaft_fixtures
    outputs = SimpleNamespace(logits=logits)

    loss = eaft_loss(outputs, labels, num_items_in_batch, alpha=alpha, k=k)
    grad = torch.autograd.grad(loss, logits)[0]
    ref = _reference(logits, labels, num_items_in_batch, alpha=alpha, k=k)
    ref_grad = torch.autograd.grad(ref, logits)[0]

    assert torch.allclose(loss, ref, atol=1e-6)
    assert torch.allclose(grad, ref_grad, atol=1e-7)
    # ignored targets contribute no gradient to the logits that predict them
    shift_ignored = labels[..., 1:] == -100
    assert torch.all(grad[..., :-1, :][shift_ignored] == 0)


def test_eaft_bf16_matches_masked_reference(eaft_fixtures):
    logits, labels = eaft_fixtures
    logits = logits.detach().bfloat16()

    loss = eaft_loss(SimpleNamespace(logits=logits), labels)

    assert loss.dtype == _reference(logits, labels).dtype
    assert torch.allclose(loss, _reference(logits, labels), rtol=1e-2)


def test_eaft_all_masked_is_zero_and_finite():
    logits = torch.randn(1, 8, 32, requires_grad=True)
    labels = torch.full((1, 8), -100)

    loss = eaft_loss(SimpleNamespace(logits=logits), labels)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.all(logits.grad == 0)


def test_eaft_non_finite_weight_on_ignored_row_is_dropped():
    logits = torch.zeros(1, 4, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0] = 1e4  # one-hot softmax: zero entropy on an ignored row
    labels = torch.tensor([[-100, -100, 3, 5]])

    loss = eaft_loss(SimpleNamespace(logits=logits), labels, alpha=-1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_eaft_does_not_read_device_values_on_the_host(eaft_fixtures):
    logits, labels = eaft_fixtures

    with HostSyncRecorder() as recorder:
        eaft_loss(SimpleNamespace(logits=logits), labels).backward()

    assert recorder.ops == []
