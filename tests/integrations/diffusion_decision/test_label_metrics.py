"""Logging-only label-set mass and excess full-vocabulary loss diagnostics."""

import math

import pytest
import torch
import torch.nn.functional as F

import axolotl.integrations.diffusion_decision.loss as loss_module
from axolotl.integrations.diffusion_decision.loss import (
    DecisionLabelExample,
    DecisionLabelQuestion,
    DistributionLabel,
    HardLabel,
    SetLabel,
    decision_label_loss,
    decision_label_loss_from_hidden,
)

LABEL_FIELDS = ("label_mass", "label_mass_nll", "excess_full_vocab_loss")
PER_EXAMPLE_LABEL_FIELDS = tuple(f"per_example_{name}" for name in LABEL_FIELDS)
LOSS_FIELDS = (
    "loss",
    "restricted_loss",
    "full_vocab_loss",
    "effective_full_vocab_loss",
    "brier_loss",
)


def _dense_linear_token_loss(hidden, head, targets):
    return F.cross_entropy(
        F.linear(hidden, head.weight, head.bias).float(), targets, reduction="none"
    )


def _question(position, ids, target, weight=1.0):
    return DecisionLabelQuestion(position, tuple(ids), target, weight)


def _example(*questions, weight=1.0):
    return DecisionLabelExample(questions=tuple(questions), source_weight=weight)


def _one(logits_row, ids, target, **kwargs):
    logits = torch.tensor([[logits_row]], dtype=torch.float32)
    return decision_label_loss(
        logits, [_example(_question(0, ids, target))], torch.tensor([[True]]), **kwargs
    )


def _entropy(values):
    return -sum(value * math.log(value) for value in values if value > 0)


ROW = [0.4, -0.3, 1.2, 0.1, -0.8, 0.6]


def _mass(row, ids):
    probs = torch.tensor(row, dtype=torch.float32).softmax(0)
    return probs[list(ids)].sum()


@pytest.mark.parametrize("mode", ["both", "full"])
def test_hard_target_excess_equals_full_vocab_ce(mode):
    result = _one(ROW, (1, 2, 4), HardLabel(1), label_softmax=mode)
    torch.testing.assert_close(result.excess_full_vocab_loss, result.full_vocab_loss)
    expected_ce = -torch.tensor(ROW).log_softmax(0)[2]
    torch.testing.assert_close(result.full_vocab_loss, expected_ce)
    torch.testing.assert_close(result.label_mass, _mass(ROW, (1, 2, 4)))
    torch.testing.assert_close(result.label_mass_nll, -_mass(ROW, (1, 2, 4)).log())
    assert 0.0 <= result.label_mass.item() <= 1.0


def test_set_target_excess_equals_full_vocab_ce_of_selected_member():
    # SetLabel full-vocab CE targets the top-scoring member (token 0): H = 0.
    result = _one(ROW, (0, 2, 3), SetLabel((0, 2)))
    torch.testing.assert_close(result.excess_full_vocab_loss, result.full_vocab_loss)
    torch.testing.assert_close(
        result.full_vocab_loss, -torch.tensor(ROW).log_softmax(0)[0]
    )


def test_soft_target_excess_is_full_ce_minus_target_entropy():
    probs = (0.2, 0.5, 0.3)
    result = _one(ROW, (1, 2, 4), DistributionLabel(probs))
    logprobs = torch.tensor(ROW).log_softmax(0)
    full_ce = -(torch.tensor(probs) * logprobs[[1, 2, 4]]).sum()
    torch.testing.assert_close(result.full_vocab_loss, full_ce)
    torch.testing.assert_close(
        result.excess_full_vocab_loss, full_ce - torch.tensor(_entropy(probs))
    )
    assert result.excess_full_vocab_loss.item() > 0.0


def test_sparse_soft_target_entropy_ignores_zero_mass():
    target = DistributionLabel((0.25, 0.75), candidate_indices=(0, 2))
    result = _one(ROW, (1, 2, 4), target)
    torch.testing.assert_close(
        result.excess_full_vocab_loss,
        result.full_vocab_loss - torch.tensor(_entropy((0.25, 0.75))),
    )


@pytest.mark.parametrize(
    "target, kwargs",
    [
        (HardLabel(0, smoothing=0.3), {}),
        (HardLabel(0), {"hard_label_smoothing": 0.3}),
        (DistributionLabel((1.0, 0.0, 0.0)), {"hard_label_smoothing": 0.3}),
    ],
    ids=["per-label-smoothing", "global-smoothing", "one-hot-dist-smoothing"],
)
def test_smoothed_hard_target_entropy_is_the_smoothed_distribution(target, kwargs):
    result = _one(ROW, (1, 2, 4), target, **kwargs)
    smoothed = (0.7 + 0.1, 0.1, 0.1)
    torch.testing.assert_close(
        result.excess_full_vocab_loss,
        result.full_vocab_loss - torch.tensor(_entropy(smoothed)),
    )


def test_other_probability_mass_is_part_of_the_target_entropy():
    target = DistributionLabel(
        (0.6, 0.3), candidate_indices=(0, 1), other_probability=0.1
    )
    result = _one(ROW, (1, 2, 4), target, label_softmax="full")
    logprobs = torch.tensor(ROW).log_softmax(0)
    other = logprobs[[0, 3, 4, 5]].logsumexp(0)
    full_ce = -(0.6 * logprobs[1] + 0.3 * logprobs[2] + 0.1 * other)
    torch.testing.assert_close(result.full_vocab_loss, full_ce)
    torch.testing.assert_close(
        result.excess_full_vocab_loss,
        full_ce - torch.tensor(_entropy((0.6, 0.3, 0.1))),
    )
    # The excess is a KL over (candidates, other bucket): zero at the optimum.
    optimum = [
        math.log(v) if v else -math.inf for v in (0.05, 0.6, 0.3, 0.025, 0, 0.025)
    ]
    at_optimum = _one(optimum, (1, 2, 4), target, label_softmax="full")
    torch.testing.assert_close(
        at_optimum.excess_full_vocab_loss, torch.tensor(0.0), atol=1e-6, rtol=0
    )


def test_label_mass_is_one_when_non_label_logits_are_negative_infinity():
    row = [-math.inf, 0.3, -0.2, -math.inf, 1.1, -math.inf]
    result = _one(row, (1, 2, 4), HardLabel(2))
    torch.testing.assert_close(result.label_mass, torch.tensor(1.0))
    torch.testing.assert_close(result.label_mass_nll, torch.tensor(0.0))
    # The full CE then equals the restricted CE, and both are the excess.
    torch.testing.assert_close(result.excess_full_vocab_loss, result.restricted_loss)


def test_label_mass_stays_in_unit_interval_for_random_logits():
    torch.manual_seed(3)
    for _ in range(20):
        row = (torch.randn(9) * 6).tolist()
        result = _one(row, (2, 5, 7), DistributionLabel((0.1, 0.6, 0.3)))
        assert 0.0 <= result.label_mass.item() <= 1.0
        assert result.label_mass_nll.item() >= 0.0
        assert result.excess_full_vocab_loss.item() >= -1e-6


def test_restricted_softmax_reports_no_label_metrics():
    result = _one(ROW, (1, 2, 4), HardLabel(1), label_softmax="restricted")
    for name in LABEL_FIELDS + PER_EXAMPLE_LABEL_FIELDS:
        assert getattr(result, name) is None


MIXED_EXAMPLES = [
    _example(_question(0, (0, 3), SetLabel((0, 1))), weight=1.75),
    _example(
        _question(1, (1, 2, 4), DistributionLabel((0.0, 0.4, 0.6)), weight=0.5),
        _question(2, (2, 5), HardLabel(0), weight=2.0),
        weight=0.25,
    ),
    _example(
        _question(0, (1, 2, 6), DistributionLabel((0.3, 0.6), (0, 2), 0.1)),
        _question(2, (0, 4), HardLabel(1, smoothing=0.2)),
    ),
]
MIXED_MASK = torch.tensor(
    [[True, False, False], [False, True, True], [True, False, True]]
)


def _manual_label_metrics(logits, examples):
    """Independent per-example reference using the same question/source weights."""
    per_example = []
    for batch_index, example in enumerate(examples):
        masses, nlls, excesses, weights = [], [], [], []
        for question in example.questions:
            row = logits[batch_index, question.position].double()
            logprobs = row.log_softmax(0)
            ids = list(question.allowed_token_ids)
            log_mass = logprobs[ids].logsumexp(0)
            target = question.target
            if isinstance(target, HardLabel):
                smoothing = target.smoothing or 0.0
                count = len(ids)
                desired = [smoothing / count] * count
                desired[target.gold_index] += 1.0 - smoothing
                other = 0.0
                full = -sum(d * logprobs[i] for d, i in zip(desired, ids, strict=True))
            elif isinstance(target, SetLabel):
                member = max(target.allowed_indices, key=lambda i: row[ids[i]])
                desired, other = [1.0], 0.0
                full = -logprobs[ids[member]]
            else:
                candidates = target.candidate_indices or range(len(ids))
                desired = list(target.probabilities)
                other = target.other_probability
                cand_ids = [ids[i] for i in candidates]
                full = -sum(
                    d * logprobs[i] for d, i in zip(desired, cand_ids, strict=True)
                )
                if other:
                    rest = [i for i in range(row.shape[0]) if i not in cand_ids]
                    full = full - other * logprobs[rest].logsumexp(0)
            masses.append(log_mass.exp())
            nlls.append(-log_mass)
            excesses.append(full - _entropy([*desired, other]))
            weights.append(question.weight)
        count = len(weights)
        scale = example.source_weight / count
        per_example.append(
            tuple(
                sum(w * v for w, v in zip(weights, values, strict=True)) * scale
                for values in (masses, nlls, excesses)
            )
        )
    return [torch.stack(values).float() for values in zip(*per_example, strict=True)]


@pytest.mark.parametrize("path", ["dense", "hidden"])
@pytest.mark.parametrize("mode", ["both", "full"])
def test_per_example_metrics_apply_question_and_source_weights(path, mode):
    examples = MIXED_EXAMPLES if mode == "full" else MIXED_EXAMPLES[:2]
    mask = MIXED_MASK[: len(examples)]
    torch.manual_seed(5)
    hidden = torch.randn(len(examples), 3, 4)
    head = torch.nn.Linear(4, 7)
    logits = F.linear(hidden, head.weight, head.bias)
    if path == "dense":
        result = decision_label_loss(logits, examples, mask, label_softmax=mode)
    else:
        result = decision_label_loss_from_hidden(
            hidden,
            head,
            examples,
            mask,
            linear_token_loss=_dense_linear_token_loss,
            label_softmax=mode,
        )
    expected = _manual_label_metrics(logits.detach(), examples)
    for name, values in zip(PER_EXAMPLE_LABEL_FIELDS, expected, strict=True):
        torch.testing.assert_close(torch.stack(getattr(result, name)), values)
    for name, values in zip(LABEL_FIELDS, expected, strict=True):
        torch.testing.assert_close(getattr(result, name), values.mean())
    for name in LABEL_FIELDS + PER_EXAMPLE_LABEL_FIELDS:
        value = getattr(result, name)
        for tensor in value if isinstance(value, tuple) else (value,):
            assert not tensor.requires_grad


def _loss_and_grads(path, examples, mask, kwargs):
    torch.manual_seed(11)
    hidden = torch.randn(len(examples), 3, 4, dtype=torch.float32)
    head = torch.nn.Linear(4, 7)
    hidden.requires_grad_()
    if path == "dense":
        logits = F.linear(hidden, head.weight, head.bias)
        result = decision_label_loss(logits, examples, mask, **kwargs)
    else:
        result = decision_label_loss_from_hidden(
            hidden,
            head,
            examples,
            mask,
            linear_token_loss=_dense_linear_token_loss,
            **kwargs,
        )
    result.loss.backward()
    return result, (hidden.grad, head.weight.grad, head.bias.grad)


@pytest.mark.parametrize("path", ["dense", "hidden"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"label_softmax": "both"},
        {"label_softmax": "both", "full_ce_weighting": "dft"},
        {"label_softmax": "both", "hard_label_smoothing": 0.1},
        {"label_softmax": "full"},
        {"label_softmax": "restricted"},
    ],
    ids=["both", "dft", "smoothing", "full", "restricted"],
)
def test_label_metrics_leave_loss_and_gradients_bit_identical(
    monkeypatch, path, kwargs
):
    examples = (
        MIXED_EXAMPLES if kwargs["label_softmax"] == "full" else MIXED_EXAMPLES[:2]
    )
    mask = MIXED_MASK[: len(examples)]
    with_metrics, grads = _loss_and_grads(path, examples, mask, kwargs)

    # Knock the diagnostics out entirely: the objective must not move by one ulp.
    monkeypatch.setattr(
        loss_module, "_label_log_mass", lambda logprobs, ids: logprobs.new_zeros(())
    )
    monkeypatch.setattr(
        loss_module,
        "_excess_full_vocab",
        lambda full_vocab, *_args: full_vocab.new_zeros(()),
    )
    monkeypatch.setattr(loss_module, "_target_entropy", lambda *_args: torch.zeros(()))
    monkeypatch.setattr(loss_module, "_label_metric_fields", lambda _values: {})
    without_metrics, baseline_grads = _loss_and_grads(path, examples, mask, kwargs)

    assert without_metrics.label_mass is None
    for name in LOSS_FIELDS:
        assert torch.equal(getattr(with_metrics, name), getattr(without_metrics, name))
    for name in (
        "per_example_loss",
        "per_example_restricted_loss",
        "per_example_full_vocab_loss",
        "per_example_brier_loss",
    ):
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                getattr(with_metrics, name), getattr(without_metrics, name), strict=True
            )
        )
    for left, right in zip(grads, baseline_grads, strict=True):
        assert torch.equal(left, right)
