import math

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.diffusion_decision.loss import (
    DecisionLabelExample,
    DecisionLabelQuestion,
    DistributionLabel,
    HardLabel,
    SetLabel,
    decision_label_loss,
    decision_label_loss_from_hidden,
    label_target_from_mapping,
)

ONE = torch.tensor([[True]])
MIXED_MASK = torch.tensor([[True, False, False], [False, True, True]])
FULL = {"label_softmax": "full"}
RESULT_FIELDS = (
    "loss",
    "restricted_loss",
    "full_vocab_loss",
    "effective_full_vocab_loss",
    "full_vocab_dft_hard_weight_sum",
    "full_vocab_dft_hard_count",
    "brier_loss",
)


def _dense_linear_token_loss(hidden, head, targets):
    return F.cross_entropy(
        F.linear(hidden, head.weight, head.bias).float(), targets, reduction="none"
    )


def _example(*questions, weight=1.0):
    return DecisionLabelExample(questions=tuple(questions), source_weight=weight)


def _question(position, ids, target):
    return DecisionLabelQuestion(position, tuple(ids), target)


def _single(position, ids, target, weight=1.0):
    return _example(_question(position, ids, target), weight=weight)


def _dist(probs, candidates=None, other=None):
    fields = {} if candidates is None else {"candidate_indices": candidates}
    if other is not None:
        fields["other_probability"] = other
    return DistributionLabel(tuple(probs), **fields)


def _row(*values, grad=False, dtype=None):
    return torch.tensor([[list(values)]], dtype=dtype, requires_grad=grad)


MIXED_EXAMPLES = [
    _single(0, (0, 3), SetLabel((0, 1)), weight=1.75),
    _example(
        _question(1, (1, 2, 4), _dist((0.0, 0.4, 0.6))),
        _question(2, (2, 5), HardLabel(0)),
        weight=0.25,
    ),
]


def _loss(logits, examples, mask=ONE, **kwargs):
    if not isinstance(examples, list):
        examples = [examples]
    return decision_label_loss(logits, examples, mask, **kwargs)


def _hidden_loss(hidden, head, examples, mask=ONE, **kwargs):
    if not isinstance(examples, list):
        examples = [examples]
    kwargs.setdefault("linear_token_loss", _dense_linear_token_loss)
    return decision_label_loss_from_hidden(hidden, head, examples, mask, **kwargs)


def _assert_fields_close(actual, expected, names=RESULT_FIELDS, **tolerance):
    for name in names:
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), **tolerance
        )


def _assert_finite_backward(result, *leaves):
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for leaf in leaves:
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


@pytest.mark.parametrize("mode", ["both", "full", "restricted"])
def test_per_example_components_average_to_aggregates_for_mixed_examples(mode):
    torch.manual_seed(41)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64)
    head = torch.nn.Linear(4, 7, dtype=torch.float64)
    result = _hidden_loss(hidden, head, MIXED_EXAMPLES, MIXED_MASK, label_softmax=mode)
    for aggregate, values in [
        (result.loss, result.per_example_loss),
        (result.restricted_loss, result.per_example_restricted_loss),
        (result.full_vocab_loss, result.per_example_full_vocab_loss),
        (result.brier_loss, result.per_example_brier_loss),
    ]:
        assert values is not None
        torch.testing.assert_close(torch.stack(values).mean(), aggregate)
        assert all(not value.requires_grad for value in values)


@pytest.mark.parametrize("mode", ["both", "full", "restricted"])
def test_hidden_loss_accepts_accelerate_fp32_outputs_with_bf16_head(mode):
    from accelerate.utils import convert_outputs_to_fp32

    torch.manual_seed(19)
    encoder = torch.nn.Linear(4, 4)
    head = torch.nn.Linear(4, 3, dtype=torch.bfloat16)

    @convert_outputs_to_fp32
    def forward(values):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return {"last_hidden_state": encoder(values)}

    hidden = forward(torch.ones(1, 1, 4))["last_hidden_state"]
    hidden.retain_grad()
    assert hidden.dtype == torch.float32
    example = _single(0, (0, 1, 2), HardLabel(1))
    seen = []

    def callback(query_hidden, callback_head, targets):
        seen.append(query_hidden.dtype)
        return _dense_linear_token_loss(query_hidden, callback_head, targets)

    result = _hidden_loss(
        hidden, head, example, linear_token_loss=callback, label_softmax=mode
    )
    reference = _loss(
        F.linear(hidden.to(head.weight.dtype), head.weight, head.bias).float(),
        example,
        label_softmax=mode,
    )
    torch.testing.assert_close(result.loss, reference.loss)
    result.loss.backward()
    assert seen == ([] if mode == "restricted" else [torch.bfloat16])
    assert hidden.grad.dtype == torch.float32
    assert torch.isfinite(hidden.grad).all() and torch.count_nonzero(hidden.grad)
    assert encoder.weight.grad is not None
    assert torch.isfinite(encoder.weight.grad).all()
    assert head.weight.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("target", "mode"),
    [
        (HardLabel(1), "both"),
        (_dist((0.0, 0.25, 0.75)), "both"),
        (SetLabel((0, 2)), "both"),
        (HardLabel(1), "full"),
        (SetLabel((0, 2)), "restricted"),
    ],
)
def test_hidden_cce_loss_matches_dense_callback_and_backpropagates(target, mode):
    torch.manual_seed(7)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(4, 7, bias=True, dtype=torch.float64)
    examples = [
        _single(0, (1, 3, 5), target, weight=2.0),
        _example(
            _question(1, (0, 2), HardLabel(0)),
            _question(2, (1, 4, 6), _dist((0.0, 0.5, 0.5))),
            weight=0.5,
        ),
    ]
    calls = []

    def callback(query_hidden, callback_head, targets):
        calls.append((query_hidden.shape, targets.shape))
        return _dense_linear_token_loss(query_hidden, callback_head, targets)

    result = _hidden_loss(
        hidden,
        head,
        examples,
        MIXED_MASK,
        linear_token_loss=callback,
        label_softmax=mode,
        brier_weight=0.3,
    )
    _assert_finite_backward(result, hidden, head.weight, head.bias)
    expected_calls = (
        [] if mode == "restricted" else [(torch.Size([3, 4]), torch.Size([3]))]
    )
    assert calls == expected_calls
    if mode == "full":
        torch.testing.assert_close(
            result.restricted_loss, torch.zeros_like(result.restricted_loss)
        )


def test_hidden_cce_helper_rejects_reduced_callback_output():
    head = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    with pytest.raises(ValueError, match="one loss per selected"):
        _hidden_loss(
            torch.zeros(1, 1, 2, dtype=torch.float64),
            head,
            _single(0, (0, 1), HardLabel(0)),
            linear_token_loss=lambda values, linear, targets: _dense_linear_token_loss(
                values, linear, targets
            ).mean(),
        )


@pytest.mark.parametrize(
    "target", [HardLabel(150), _dist((0.0,) * 150 + (1.0,))], ids=["hard", "dist"]
)
def test_151_candidate_final_index_has_normalized_loss_and_gradients(target):
    hidden = torch.randn(1, 1, 4, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(4, 151, bias=True, dtype=torch.float64)
    logits = F.linear(hidden, head.weight, head.bias)
    result = _loss(logits, _single(0, tuple(range(151)), target))
    _assert_finite_backward(result, hidden, head.weight)


@pytest.mark.parametrize(
    ("examples", "mask", "kwargs", "fields", "dtype"),
    [
        (MIXED_EXAMPLES, MIXED_MASK, {}, RESULT_FIELDS, torch.float64),
        (
            MIXED_EXAMPLES,
            MIXED_MASK,
            {"full_ce_weighting": "dft"},
            RESULT_FIELDS,
            torch.float64,
        ),
        (
            _single(0, (0, 3), _dist((0.0, 1.0))),
            ONE,
            {**FULL, "hard_label_smoothing": 0.1},
            ("loss", "full_vocab_loss", "brier_loss"),
            torch.float64,
        ),
        (
            _single(0, (0, 1), _dist((0.5,), (1,), 0.5)),
            ONE,
            {**FULL, "brier_weight": 0.0},
            ("loss", "full_vocab_loss"),
            torch.float32,
        ),
    ],
    ids=[
        "ce-typed-unequal-questions",
        "dft-typed-unequal-questions",
        "one-hot-distribution-smoothing",
        "sparse-distribution-residual",
    ],
)
def test_hidden_path_matches_dense_loss_and_gradients(
    examples, mask, kwargs, fields, dtype
):
    torch.manual_seed(11)
    hidden = torch.randn(*mask.shape, 4, dtype=dtype, requires_grad=True)
    head = torch.nn.Linear(4, 7, dtype=dtype)
    dense_hidden = hidden.detach().clone().requires_grad_()
    dense_head = torch.nn.Linear(4, 7, dtype=dtype)
    dense_head.load_state_dict(head.state_dict())
    cce = _hidden_loss(hidden, head, examples, mask, **kwargs)
    logits = F.linear(dense_hidden, dense_head.weight, dense_head.bias)
    dense = _loss(logits, examples, mask, **kwargs)
    _assert_fields_close(cce, dense, fields)
    cce.loss.backward()
    dense.loss.backward()
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(head.weight.grad, dense_head.weight.grad)
    torch.testing.assert_close(head.bias.grad, dense_head.bias.grad)


@pytest.mark.parametrize(
    ("examples", "mask", "variant"),
    [
        (
            [_single(0, (0, 2, 3), HardLabel(1)), _single(1, (1, 3), SetLabel((0, 1)))],
            torch.tensor([[True, False], [False, True]]),
            "ce",
        ),
        (
            [
                _example(
                    _question(0, (0, 1, 2), _dist((0.2, 0.3, 0.5))),
                    _question(1, (0, 2, 3), SetLabel((0, 2))),
                ),
                _single(0, (1, 2, 3), _dist((0.5, 0.5, 0.0))),
            ],
            torch.tensor([[True, True], [True, False]]),
            "dft",
        ),
    ],
    ids=["explicit-ce-is-default", "dft-leaves-distribution-and-set-unchanged"],
)
def test_full_ce_weighting_variants_preserve_loss_and_gradients(
    examples, mask, variant
):
    logits = torch.tensor(
        [
            [[0.4, -0.2, 1.3, 0.1], [0.8, -0.1, 0.2, 1.1]],
            [[0.3, 0.9, -0.6, 0.2], [-0.4, 0.5, 1.0, 0.7]],
        ],
        dtype=torch.float64,
    )
    baseline_logits = logits.clone().requires_grad_()
    variant_logits = logits.clone().requires_grad_()
    baseline = _loss(baseline_logits, examples, mask)
    result = _loss(variant_logits, examples, mask, full_ce_weighting=variant)
    baseline.loss.backward()
    result.loss.backward()

    torch.testing.assert_close(result.loss, baseline.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        variant_logits.grad, baseline_logits.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.effective_full_vocab_loss, result.full_vocab_loss, rtol=0, atol=0
    )
    zero = torch.zeros_like(result.loss)
    torch.testing.assert_close(result.full_vocab_dft_hard_weight_sum, zero)
    torch.testing.assert_close(result.full_vocab_dft_hard_count, zero)


def test_dft_full_ce_uses_detached_target_probability_and_preserves_raw_metric():
    logits = _row(0.3, -0.5, 1.1, grad=True, dtype=torch.float64)
    expected_logits = logits.detach().clone().requires_grad_()
    result = _loss(
        logits,
        _single(0, (0, 1, 2), HardLabel(2)),
        full_ce_weighting="dft",
        brier_weight=0,
        **FULL,
    )
    full_ce = -F.log_softmax(logits[0, 0].float(), dim=0)[2]
    probability = full_ce.detach().neg().exp()
    torch.testing.assert_close(result.full_vocab_loss, full_ce)
    torch.testing.assert_close(result.effective_full_vocab_loss, probability * full_ce)
    torch.testing.assert_close(result.full_vocab_dft_hard_weight_sum, probability)
    torch.testing.assert_close(
        result.full_vocab_dft_hard_count, torch.ones_like(probability)
    )
    result.loss.backward()
    (-F.softmax(expected_logits[0, 0].float(), dim=0)[2]).backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad)


def _nll(row, index, ids=None):
    return -F.log_softmax(row if ids is None else row[list(ids)], dim=0)[index]


def _hard_components(row):
    restricted, full = _nll(row, 1, (0, 3)), _nll(row, 3)
    brier = (F.softmax(row[[0, 3]], dim=0) - torch.tensor([0.0, 1.0])).square().sum()
    return {
        "restricted_loss": restricted,
        "full_vocab_loss": full,
        "brier_loss": brier,
        "loss": restricted + full + 0.5 * brier,
    }


def _distribution_components(row):
    target = torch.tensor([0.25, 0.75])
    kl = torch.sum(target * (target.log() - F.log_softmax(row[[1, 2]], dim=0)))
    full = -(target * F.log_softmax(row, dim=0)[[1, 2]]).sum()
    return {"restricted_loss": kl, "full_vocab_loss": full, "loss": kl + full}


def _sparse_residual_components(row):
    logprobs = F.log_softmax(row, dim=0)
    full = -(torch.tensor([0.25, 0.5]) * logprobs[[1, 2]]).sum()
    full = full - 0.25 * torch.log1p(-logprobs[[1, 2]].exp().sum())
    return {"full_vocab_loss": full, "loss": full}


def _set_components(row):
    mass = F.softmax(row[[0, 1, 2]], dim=0)[[0, 2]].sum()
    full = _nll(row, 0)
    return {
        "restricted_loss": -mass.log(),
        "full_vocab_loss": full,
        "brier_loss": (1.0 - mass).square(),
        "loss": -mass.log() + full + 0.25 * (1.0 - mass).square(),
    }


ZERO = torch.zeros(())


@pytest.mark.parametrize(
    ("logits", "question", "kwargs", "expected"),
    [
        (
            torch.tensor([[[30.0, -30.0, 20.0, -20.0], [0.2, 1.5, -0.4, 0.7]]]),
            _question(1, (0, 3), HardLabel(1)),
            {"brier_weight": 0.5},
            _hard_components,
        ),
        (
            _row(0.3, -0.2, 1.1, 0.0),
            _question(0, (1, 2), _dist((0.25, 0.75))),
            {"brier_weight": 0.0},
            _distribution_components,
        ),
        (
            _row(0.3, -0.2, 1.1, 0.0),
            _question(0, (0, 1, 2), _dist((0.25, 0.5), (1, 2), 0.25)),
            {**FULL, "brier_weight": 0.0},
            _sparse_residual_components,
        ),
        (
            _row(0.1, 0.2, 1.5),
            _question(0, (0, 1, 2), _dist((0.0, 0.0, 1.0))),
            {"brier_weight": 0},
            lambda row: {"restricted_loss": _nll(row, 2)},
        ),
        (
            _row(1.0, -1.0, 0.4, 2.0),
            _question(0, (0, 1, 2), SetLabel((0, 2))),
            {"brier_weight": 0.25},
            _set_components,
        ),
        (
            _row(0.3, 1.2, -0.1),
            _question(0, (0, 1, 2), SetLabel((0, 1, 2))),
            {"label_softmax": "restricted", "brier_weight": 0},
            lambda row: {"restricted_loss": ZERO, "brier_loss": ZERO},
        ),
        (
            _row(0.1, 1.2, -0.2),
            _question(0, (0, 2), HardLabel(0)),
            {**FULL, "brier_weight": 0},
            lambda row: {"restricted_loss": ZERO, "full_vocab_loss": _nll(row, 0)},
        ),
    ],
    ids=[
        "hard-combines-restricted-full-and-brier-at-supervised-position",
        "distribution-restricted-kl-and-full-soft-ce",
        "sparse-distribution-full-ce-keeps-residual-other-mass",
        "distribution-kl-accepts-zero-probability-alternatives",
        "set-allowed-mass-and-detached-best-member-full-ce",
        "set-mass-zero-when-every-allowed-label-correct",
        "full-mode-excludes-restricted-primary-for-hard",
    ],
)
def test_single_question_loss_components(logits, question, kwargs, expected):
    mask = torch.zeros(logits.shape[:2], dtype=torch.bool)
    mask[0, question.position] = True
    result = _loss(logits, _example(question), mask, **kwargs)
    assert torch.isfinite(result.loss)
    for name, value in expected(logits[0, question.position]).items():
        torch.testing.assert_close(getattr(result, name), value)


@pytest.mark.parametrize(
    ("allowed_ids", "target", "expected_target", "gold", "hard"),
    [
        ((0, 3), _dist((0.0, 1.0)), (0.05, 0.95), 3, (0.0, 1.0)),
        (
            (0, 1, 2, 3),
            _dist((0.0, 0.0, 1.0, 0.0)),
            (0.025, 0.025, 0.925, 0.025),
            2,
            (0.0, 0.0, 1.0, 0.0),
        ),
        ((0, 1, 2, 3), _dist((1.0,), (3,)), (0.025, 0.025, 0.025, 0.925), 3, None),
    ],
    ids=["two-way", "four-way", "sparse-one-hot"],
)
def test_one_hot_distribution_smoothing_uses_allowed_support_for_ce_and_keeps_hard_brier(
    allowed_ids, target, expected_target, gold, hard
):
    logits = _row(0.2, 1.5, -0.4, 0.7, 1.1, grad=True)
    example = _single(0, allowed_ids, target)
    result = _loss(logits, example, brier_weight=0.5, hard_label_smoothing=0.1, **FULL)
    unsmoothed = _loss(logits, example, brier_weight=0.5, **FULL)
    result.loss.backward()

    full_logprobs = F.log_softmax(logits.detach()[0, 0], dim=0)
    torch.testing.assert_close(
        result.full_vocab_loss,
        -(torch.tensor(expected_target) * full_logprobs[list(allowed_ids)]).sum(),
    )
    if hard is not None:
        restricted_probs = F.softmax(logits.detach()[0, 0, list(allowed_ids)], dim=0)
        torch.testing.assert_close(
            result.brier_loss, (restricted_probs - torch.tensor(hard)).square().sum()
        )
    assert not torch.allclose(result.loss, unsmoothed.loss)
    assert logits.grad is not None
    assert logits.grad[0, 0, gold] < 0
    for token_id in set(range(logits.shape[-1])) - set(allowed_ids):
        assert logits.grad[0, 0, token_id] > 0


@pytest.mark.parametrize(
    ("logits", "examples", "kwargs"),
    [
        (
            _row(0.0, 100.0, 0.0, 0.0, 0.0, grad=True),
            _single(0, (0, 1), _dist((0.5,), (1,), 0.5)),
            {**FULL, "brier_weight": 0.0},
        ),
        (
            _row(1000.0, -1000.0, 999.0, grad=True),
            _single(0, (0, 1, 2), SetLabel((1,))),
            {"label_softmax": "restricted", "brier_weight": 0},
        ),
        (
            torch.tensor(
                [[[0.1, 0.4, -0.3]], [[0.1, 0.4, -0.3]]],
                dtype=torch.float64,
                requires_grad=True,
            ),
            [
                _single(0, (0, 1, 2), HardLabel(1)),
                _single(0, (0, 1, 2), HardLabel(1, smoothing=0.1)),
            ],
            {**FULL, "hard_label_smoothing": 0.0},
        ),
    ],
    ids=[
        "sparse-residual-when-candidate-mass-rounds-to-one",
        "set-mass-logsumexp-for-extremely-unlikely-correct-label",
        "mixed-per-target-smoothing-with-global-smoothing-disabled",
    ],
)
def test_loss_and_gradients_stay_finite_at_numerical_edges(logits, examples, kwargs):
    mask = torch.ones(logits.shape[:2], dtype=torch.bool)
    _assert_finite_backward(_loss(logits, examples, mask, **kwargs), logits)


def test_distribution_full_ce_has_soft_target_gradient_and_ignores_zero_mass():
    logits = _row(0.7, -0.3, 1.2, 0.1, grad=True)
    example = _single(0, (0, 2, 3), _dist((0.25, 0.75, 0.0)))
    result = _loss(logits, example, brier_weight=0.0, **FULL)
    result.loss.backward()

    target = torch.tensor([0.25, 0.75, 0.0])
    expected = -(target * F.log_softmax(logits.detach()[0, 0], 0)[[0, 2, 3]]).sum()
    torch.testing.assert_close(result.full_vocab_loss, expected)
    assert logits.grad is not None
    assert logits.grad[0, 0, 0] != 0
    assert logits.grad[0, 0, 2] != 0


def test_question_then_example_average_applies_source_weights():
    examples = [
        _example(
            _question(0, (0, 1), HardLabel(0)),
            _question(1, (2, 3), HardLabel(1)),
            weight=2.0,
        ),
        _single(0, (0, 1, 2, 3), HardLabel(2), weight=0.5),
    ]
    result = _loss(
        torch.zeros(2, 2, 4),
        examples,
        torch.tensor([[True, True], [True, False]]),
        label_softmax="restricted",
        brier_weight=0,
    )
    expected = (2.0 * math.log(2) + 0.5 * math.log(4)) / 2
    torch.testing.assert_close(result.loss, torch.tensor(expected))


def test_loss_only_backpropagates_through_supervised_label_positions():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    result = _loss(logits, _single(1, (1, 3), HardLabel(0)), mask, brier_weight=0)
    result.loss.backward()

    assert logits.grad is not None
    assert not torch.count_nonzero(logits.grad[0, 0])
    assert torch.count_nonzero(logits.grad[0, 1])
    assert not torch.count_nonzero(logits.grad[0, 2])


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ({"kind": "hard", "gold_idx": -1}, "gold_idx"),
        ({"kind": "dist", "probs": [0.2, 0.2]}, "sum to one"),
        ({"kind": "set", "allowed_set": [0, 0]}, "unique"),
        ({"kind": "other"}, "unknown"),
    ],
)
def test_mapping_target_contracts(target, error):
    with pytest.raises(ValueError, match=error):
        label_target_from_mapping(target)


@pytest.mark.parametrize(
    ("example", "kwargs", "error", "match"),
    [
        (_single(1, (0, 1), _dist((0.5, 0.5))), {}, RuntimeError, "supervised"),
        (_single(0, (0, 1), _dist((1.0,))), {}, ValueError, "align"),
        (
            _example(
                _question(0, (0, 1), HardLabel(0)), _question(0, (1, 2), HardLabel(1))
            ),
            {},
            ValueError,
            "distinct",
        ),
        (
            _single(0, (0, 1), _dist((0.75,), (1,), 0.25)),
            {"label_softmax": "both"},
            ValueError,
            "requires label_softmax=full",
        ),
        (_single(True, (0, 1), HardLabel(0)), {}, ValueError, None),
        (_single(0, (True, 1), HardLabel(0)), {}, ValueError, None),
        (_single(0, (0, 1), HardLabel(True)), {}, ValueError, None),
        (_single(0, (0, 1), SetLabel((True,))), {}, ValueError, None),
    ],
    ids=[
        "unsupervised-label-position",
        "misaligned-distribution-target",
        "duplicate-question-positions",
        "sparse-residual-needs-full-softmax",
        "boolean-position",
        "boolean-allowed-id",
        "boolean-gold-index",
        "boolean-set-member",
    ],
)
def test_loss_rejects_invalid_examples(example, kwargs, error, match):
    with pytest.raises(error, match=match):
        _loss(torch.zeros(1, 2, 4), example, torch.tensor([[True, False]]), **kwargs)


@pytest.mark.parametrize(
    ("left", "right", "fields", "exact"),
    [
        (
            (_single(0, (0, 3), _dist((0.25, 0.75))), {**FULL, "brier_weight": 0.1}),
            (
                _single(0, (0, 3), _dist((0.25, 0.75))),
                {**FULL, "brier_weight": 0.1, "hard_label_smoothing": 0.1},
            ),
            ("loss",),
            False,
        ),
        (
            (_single(0, (0, 3), _dist((0.0, 1.0))), FULL),
            (
                _single(0, (0, 3), _dist((0.0, 1.0))),
                {**FULL, "hard_label_smoothing": 0.0},
            ),
            ("loss",),
            False,
        ),
        (
            (
                _single(0, (0, 2, 4), HardLabel(1)),
                {**FULL, "brier_weight": 0.1, "hard_label_smoothing": 0.1},
            ),
            (
                _single(0, (0, 2, 4), HardLabel(1, smoothing=0.1)),
                {**FULL, "brier_weight": 0.1, "hard_label_smoothing": 0.0},
            ),
            ("loss", "full_vocab_loss", "brier_loss"),
            True,
        ),
        (
            (_single(0, (1, 2), _dist((0.0, 1.0))), {"brier_weight": 0.0}),
            (_single(0, (1, 2), HardLabel(1)), {"brier_weight": 0.0}),
            ("loss", "full_vocab_loss"),
            False,
        ),
    ],
    ids=[
        "hard-smoothing-leaves-soft-distribution-unchanged",
        "zero-hard-smoothing-preserves-one-hot-distribution",
        "per-target-smoothing-matches-global-and-keeps-one-hot-brier",
        "one-hot-distribution-full-ce-matches-hard-label",
    ],
)
def test_equivalent_objectives_produce_the_same_loss(left, right, fields, exact):
    logits = _row(0.2, -0.7, 1.3, 0.4, 0.9, dtype=torch.float64)
    tolerance = {"rtol": 0, "atol": 0} if exact else {}
    _assert_fields_close(
        _loss(logits, left[0], **left[1]),
        _loss(logits, right[0], **right[1]),
        fields,
        **tolerance,
    )
