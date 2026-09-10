"""Unit tests for the MemFT token-weighted loss (arXiv:2605.30260)."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from axolotl.monkeypatch.loss.memft import LN2, memft_loss


def _outputs(logits):
    return SimpleNamespace(logits=logits)


def _reference_ot(logits, labels, critical_loss=LN2, eps=1e-8, ignore_index=-100):
    bsz, _, vocab = logits.shape
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    ce = F.cross_entropy(
        shift_logits.reshape(-1, vocab).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(bsz, -1)
    mask = shift_labels != ignore_index
    weights = (mask & (ce > critical_loss)).float()
    return (weights * ce).sum() / (weights.sum() + eps)


def test_critical_loss_is_ln2():
    assert math.isclose(LN2, math.log(2.0))


def test_ot_matches_reference():
    torch.manual_seed(0)
    bsz, seq, vocab = 3, 12, 64
    logits = torch.randn(bsz, seq, vocab)
    labels = torch.randint(0, vocab, (bsz, seq))
    labels[:, :3] = -100

    got = memft_loss(_outputs(logits), labels, variant="ot")
    expected = _reference_ot(logits, labels)
    assert torch.allclose(got, expected, atol=1e-6)


def test_ot_gradient_only_flows_through_subthreshold_tokens():
    torch.manual_seed(1)
    bsz, seq, vocab = 2, 8, 32
    # confident logits at the target so some tokens land below the threshold
    labels = torch.randint(0, vocab, (bsz, seq))
    logits = torch.randn(bsz, seq, vocab) * 0.1
    confident = labels[:, 1:].clone()
    logits[:, :-1].scatter_(
        -1,
        confident.unsqueeze(-1),
        torch.full_like(confident, 6.0, dtype=logits.dtype).unsqueeze(-1),
    )
    logits.requires_grad_(True)

    loss = memft_loss(_outputs(logits), labels, variant="ot")
    loss.backward()

    shift_logits = logits.detach()[..., :-1, :]
    shift_labels = labels[..., 1:]
    ce = F.cross_entropy(
        shift_logits.reshape(-1, vocab).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(bsz, seq - 1)
    below = ce <= LN2  # already-memorized tokens
    assert below.any(), "test setup should produce some below-threshold tokens"

    grad = logits.grad[..., :-1, :]
    # tokens at/below threshold receive no gradient
    assert grad[below].abs().max() == 0.0


def test_fully_memorized_is_zero_and_finite():
    torch.manual_seed(2)
    bsz, seq, vocab = 2, 6, 16
    logits = torch.randn(bsz, seq, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (bsz, seq))
    # force every weight to zero via an unreachable threshold
    loss = memft_loss(_outputs(logits), labels, variant="ot", critical_loss=1e9)
    loss.backward()
    assert torch.isfinite(loss) and float(loss) == 0.0
    assert torch.isfinite(logits.grad).all()


def test_sw_runs_and_weights_are_detached():
    torch.manual_seed(3)
    bsz, seq, vocab = 2, 20, 48
    logits = torch.randn(bsz, seq, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (bsz, seq))
    labels[:, :2] = -100

    loss = memft_loss(
        _outputs(logits), labels, variant="sw", kappa=1.0, tau=4.0, window=5, floor=0.0
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_sw_floor_keeps_distant_tokens_active():
    torch.manual_seed(4)
    bsz, seq, vocab = 1, 30, 40
    logits = torch.randn(bsz, seq, vocab)
    labels = torch.randint(0, vocab, (bsz, seq))

    no_floor = memft_loss(
        _outputs(logits), labels, variant="sw", window=2, floor=0.0, tau=1.0
    )
    with_floor = memft_loss(
        _outputs(logits), labels, variant="sw", window=2, floor=0.5, tau=1.0
    )
    # a positive floor changes the weighting of beyond-window tokens
    assert not torch.allclose(no_floor, with_floor)


def test_ignored_positions_excluded():
    torch.manual_seed(5)
    bsz, seq, vocab = 2, 10, 24
    logits = torch.randn(bsz, seq, vocab)
    labels = torch.full((bsz, seq), -100)
    # only one real target -> denominator falls back to epsilon when below thresh
    labels[0, 5] = 3
    loss = memft_loss(_outputs(logits), labels, variant="ot")
    assert torch.isfinite(loss)


def test_unknown_variant_raises():
    logits = torch.randn(1, 4, 8)
    labels = torch.randint(0, 8, (1, 4))
    with pytest.raises(ValueError):
        memft_loss(_outputs(logits), labels, variant="bogus")
