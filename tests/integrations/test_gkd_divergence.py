"""Tests for the GKD Axis A divergence seam."""

import os

import pytest
import torch

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from axolotl.integrations.gkd.divergence import (  # noqa: E402
    DIVERGENCE_REGISTRY,
    generalized_jsd_loss,
    resolve_divergence,
)

VOCAB = 32
SEQ = 5
BSZ = 2


def _batch(seed=0):
    torch.manual_seed(seed)
    student = torch.randn(BSZ, SEQ, VOCAB)
    teacher = torch.randn(BSZ, SEQ, VOCAB)
    labels = torch.full((BSZ, SEQ), -100)
    labels[:, 3:] = torch.randint(0, VOCAB, (BSZ, 2))
    return student, teacher, labels


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_matches_trl_reference(beta):
    gkd = pytest.importorskip("trl.experimental.gkd")
    student, teacher, labels = _batch()
    ours = generalized_jsd_loss(
        student, teacher, labels=labels, beta=beta, temperature=0.9
    )
    ref = gkd.GKDTrainer.generalized_jsd_loss(
        student, teacher, labels=labels, beta=beta, temperature=0.9
    )
    assert torch.allclose(ours, ref, atol=1e-6)


def test_num_items_in_batch_matches_trl():
    gkd = pytest.importorskip("trl.experimental.gkd")
    student, teacher, labels = _batch()
    ours = generalized_jsd_loss(
        student, teacher, labels=labels, beta=0.5, num_items_in_batch=4
    )
    ref = gkd.GKDTrainer.generalized_jsd_loss(
        student, teacher, labels=labels, beta=0.5, num_items_in_batch=4
    )
    assert torch.allclose(ours, ref, atol=1e-6)


def test_identical_logits_zero_loss():
    torch.manual_seed(1)
    logits = torch.randn(BSZ, SEQ, VOCAB)
    labels = torch.zeros(BSZ, SEQ, dtype=torch.long)
    for beta in (0.0, 0.5, 1.0):
        loss = generalized_jsd_loss(logits, logits.clone(), labels=labels, beta=beta)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_fully_masked_batch_no_nan():
    student, teacher, _ = _batch()
    labels = torch.full((BSZ, SEQ), -100)
    loss = generalized_jsd_loss(student, teacher, labels=labels, beta=0.5)
    assert torch.isfinite(loss).all()
    assert torch.isclose(loss, torch.tensor(0.0))


def test_resolve_forward_and_reverse_kl():
    student, teacher, labels = _batch()
    # beta arg is ignored by the fkl/rkl entries: they pin beta to 0/1.
    fkl = resolve_divergence("fkl", beta=0.7)(student, teacher, labels=labels)
    rkl = resolve_divergence("rkl", beta=0.3)(student, teacher, labels=labels)
    assert torch.allclose(
        fkl, generalized_jsd_loss(student, teacher, labels=labels, beta=0.0)
    )
    assert torch.allclose(
        rkl, generalized_jsd_loss(student, teacher, labels=labels, beta=1.0)
    )


def test_resolve_default_uses_beta():
    student, teacher, labels = _batch()
    got = resolve_divergence(None, beta=0.4)(student, teacher, labels=labels)
    assert torch.allclose(
        got, generalized_jsd_loss(student, teacher, labels=labels, beta=0.4)
    )


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown gkd_divergence"):
        resolve_divergence("not_a_divergence", beta=0.5)


def test_registry_aliases_present():
    for name in ("jsd", "forward_kl", "fkl", "reverse_kl", "rkl"):
        assert name in DIVERGENCE_REGISTRY
