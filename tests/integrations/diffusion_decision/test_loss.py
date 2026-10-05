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


def _dense_linear_token_loss(hidden, head, targets):
    return F.cross_entropy(
        F.linear(hidden, head.weight, head.bias).float(), targets, reduction="none"
    )


@pytest.mark.parametrize("mode", ["both", "full", "restricted"])
def test_per_example_components_average_to_aggregates_for_mixed_examples(mode):
    torch.manual_seed(41)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64)
    head = torch.nn.Linear(4, 5, dtype=torch.float64)
    examples = [
        _example(_question(0, (0, 1, 2), HardLabel(1)), weight=2.0),
        _example(
            _question(1, (0, 2, 3), DistributionLabel((0.2, 0.3, 0.5))),
            _question(2, (1, 3, 4), SetLabel((0, 2))),
            weight=0.5,
        ),
    ]
    mask = torch.tensor([[True, False, False], [False, True, True]])
    result = decision_label_loss_from_hidden(
        hidden,
        head,
        examples,
        mask,
        linear_token_loss=_dense_linear_token_loss,
        label_softmax=mode,
    )
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
    examples = [_example(_question(0, (0, 1, 2), HardLabel(1)))]
    mask = torch.tensor([[True]])
    seen = []

    def callback(query_hidden, callback_head, targets):
        seen.append(query_hidden.dtype)
        return _dense_linear_token_loss(query_hidden, callback_head, targets)

    result = decision_label_loss_from_hidden(
        hidden,
        head,
        examples,
        mask,
        linear_token_loss=callback,
        label_softmax=mode,
    )
    reference = decision_label_loss(
        F.linear(hidden.to(head.weight.dtype), head.weight, head.bias).float(),
        examples,
        mask,
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
        (DistributionLabel((0.0, 0.25, 0.75)), "both"),
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
        _example(_question(0, (1, 3, 5), target), weight=2.0),
        _example(
            _question(1, (0, 2), HardLabel(0)),
            _question(2, (1, 4, 6), DistributionLabel((0.0, 0.5, 0.5))),
            weight=0.5,
        ),
    ]
    mask = torch.tensor([[True, False, False], [False, True, True]])
    calls = []

    def callback(query_hidden, callback_head, targets):
        calls.append((query_hidden.shape, targets.shape))
        return _dense_linear_token_loss(query_hidden, callback_head, targets)

    result = decision_label_loss_from_hidden(
        hidden,
        head,
        examples,
        mask,
        linear_token_loss=callback,
        label_softmax=mode,
        brier_weight=0.3,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert head.bias.grad is not None and torch.isfinite(head.bias.grad).all()
    if mode == "restricted":
        assert calls == []
    else:
        assert calls == [(torch.Size([3, 4]), torch.Size([3]))]
    if mode == "full":
        torch.testing.assert_close(
            result.restricted_loss, torch.zeros_like(result.restricted_loss)
        )


def test_hidden_cce_helper_rejects_reduced_callback_output():
    hidden = torch.zeros(1, 1, 2, dtype=torch.float64)
    head = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    with pytest.raises(ValueError, match="one loss per selected"):
        decision_label_loss_from_hidden(
            hidden,
            head,
            [_example(_question(0, (0, 1), HardLabel(0)))],
            torch.tensor([[True]]),
            linear_token_loss=lambda values, linear, targets: _dense_linear_token_loss(
                values, linear, targets
            ).mean(),
        )


@pytest.mark.parametrize(
    "target",
    [HardLabel(150), DistributionLabel((0.0,) * 150 + (1.0,))],
)
def test_151_candidate_final_index_has_normalized_loss_and_gradients(target):
    hidden = torch.randn(1, 1, 4, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(4, 151, bias=True, dtype=torch.float64)
    result = decision_label_loss(
        F.linear(hidden, head.weight, head.bias),
        [_example(_question(0, tuple(range(151)), target))],
        torch.tensor([[True]]),
        brier_weight=0.1,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_hidden_cce_matches_dense_loss_and_gradients_for_typed_unequal_questions():
    torch.manual_seed(11)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    dense_hidden = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(4, 7, bias=True, dtype=torch.float64)
    dense_head = torch.nn.Linear(4, 7, bias=True, dtype=torch.float64)
    dense_head.weight.data.copy_(head.weight.data)
    dense_head.bias.data.copy_(head.bias.data)
    examples = [
        _example(_question(0, (0, 3), SetLabel((0, 1))), weight=1.75),
        _example(
            _question(1, (1, 2, 4), DistributionLabel((0.0, 0.4, 0.6))),
            _question(2, (2, 5), HardLabel(0)),
            weight=0.25,
        ),
    ]
    mask = torch.tensor([[True, False, False], [False, True, True]])
    cce = decision_label_loss_from_hidden(
        hidden,
        head,
        examples,
        mask,
        linear_token_loss=_dense_linear_token_loss,
        brier_weight=0.1,
    )
    dense = decision_label_loss(
        F.linear(dense_hidden, dense_head.weight, dense_head.bias),
        examples,
        mask,
        brier_weight=0.1,
    )
    torch.testing.assert_close(cce.loss, dense.loss)
    torch.testing.assert_close(cce.restricted_loss, dense.restricted_loss)
    torch.testing.assert_close(cce.full_vocab_loss, dense.full_vocab_loss)
    torch.testing.assert_close(cce.brier_loss, dense.brier_loss)
    cce.loss.backward()
    dense.loss.backward()
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(head.weight.grad, dense_head.weight.grad)
    torch.testing.assert_close(head.bias.grad, dense_head.bias.grad)


def test_default_full_ce_weighting_preserves_loss_and_gradients():
    torch.manual_seed(31)
    baseline_logits = torch.randn(2, 2, 5, dtype=torch.float64, requires_grad=True)
    default_logits = baseline_logits.detach().clone().requires_grad_()
    examples = [
        _example(_question(0, (0, 2, 4), HardLabel(1))),
        _example(_question(1, (1, 3), SetLabel((0, 1)))),
    ]
    mask = torch.tensor([[True, False], [False, True]])

    baseline = decision_label_loss(
        baseline_logits, examples, mask, label_softmax="both", brier_weight=0.1
    )
    default = decision_label_loss(
        default_logits,
        examples,
        mask,
        label_softmax="both",
        full_ce_weighting="ce",
        brier_weight=0.1,
    )
    baseline.loss.backward()
    default.loss.backward()

    torch.testing.assert_close(default.loss, baseline.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        default_logits.grad, baseline_logits.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        default.effective_full_vocab_loss, default.full_vocab_loss, rtol=0, atol=0
    )


def test_dft_full_ce_uses_detached_target_probability_and_preserves_raw_metric():
    logits = torch.tensor([[[0.3, -0.5, 1.1]]], dtype=torch.float64, requires_grad=True)
    expected_logits = logits.detach().clone().requires_grad_()
    examples = [_example(_question(0, (0, 1, 2), HardLabel(2)))]
    mask = torch.tensor([[True]])

    result = decision_label_loss(
        logits,
        examples,
        mask,
        label_softmax="full",
        full_ce_weighting="dft",
        brier_weight=0,
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


def test_dft_leaves_distribution_and_set_full_ce_unchanged():
    baseline_logits = torch.tensor(
        [[[0.4, -0.2, 1.3, 0.1], [0.8, -0.1, 0.2, 1.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    dft_logits = baseline_logits.detach().clone().requires_grad_()
    examples = [
        _example(
            _question(0, (0, 1, 2), DistributionLabel((0.2, 0.3, 0.5))),
            _question(1, (0, 2, 3), SetLabel((0, 2))),
        )
    ]
    mask = torch.tensor([[True, True]])
    baseline = decision_label_loss(
        baseline_logits, examples, mask, full_ce_weighting="ce"
    )
    dft = decision_label_loss(dft_logits, examples, mask, full_ce_weighting="dft")
    baseline.loss.backward()
    dft.loss.backward()

    torch.testing.assert_close(dft.loss, baseline.loss, rtol=0, atol=0)
    torch.testing.assert_close(dft_logits.grad, baseline_logits.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        dft.full_vocab_dft_hard_weight_sum, torch.zeros_like(dft.loss)
    )
    torch.testing.assert_close(
        dft.full_vocab_dft_hard_count, torch.zeros_like(dft.loss)
    )


def test_dft_hidden_path_matches_dense_for_mixed_target_semantics():
    torch.manual_seed(47)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    dense_hidden = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(4, 7, dtype=torch.float64)
    dense_head = torch.nn.Linear(4, 7, dtype=torch.float64)
    dense_head.load_state_dict(head.state_dict())
    examples = [
        _example(_question(0, (0, 3), HardLabel(1)), weight=1.25),
        _example(
            _question(1, (1, 2, 4), DistributionLabel((0.1, 0.3, 0.6))),
            _question(2, (0, 5, 6), SetLabel((0, 2))),
            weight=0.5,
        ),
    ]
    mask = torch.tensor([[True, False, False], [False, True, True]])

    hidden_result = decision_label_loss_from_hidden(
        hidden,
        head,
        examples,
        mask,
        linear_token_loss=_dense_linear_token_loss,
        full_ce_weighting="dft",
        brier_weight=0.1,
    )
    dense_result = decision_label_loss(
        F.linear(dense_hidden, dense_head.weight, dense_head.bias),
        examples,
        mask,
        full_ce_weighting="dft",
        brier_weight=0.1,
    )
    for name in (
        "loss",
        "restricted_loss",
        "full_vocab_loss",
        "effective_full_vocab_loss",
        "full_vocab_dft_hard_weight_sum",
        "full_vocab_dft_hard_count",
        "brier_loss",
    ):
        torch.testing.assert_close(
            getattr(hidden_result, name), getattr(dense_result, name)
        )
    hidden_result.loss.backward()
    dense_result.loss.backward()
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(head.weight.grad, dense_head.weight.grad)
    torch.testing.assert_close(head.bias.grad, dense_head.bias.grad)


def _example(*questions, weight=1.0):
    return DecisionLabelExample(questions=tuple(questions), source_weight=weight)


def _question(position, ids, target):
    return DecisionLabelQuestion(position, tuple(ids), target)


def test_hard_label_combines_restricted_full_vocab_and_brier_at_supervised_position():
    logits = torch.tensor([[[30.0, -30.0, 20.0, -20.0], [0.2, 1.5, -0.4, 0.7]]])
    mask = torch.tensor([[False, True]])
    example = _example(_question(1, (0, 3), HardLabel(1)))

    result = decision_label_loss(logits, [example], mask, brier_weight=0.5)

    row = logits[0, 1]
    restricted = -F.log_softmax(row[[0, 3]], dim=0)[1]
    full = -F.log_softmax(row, dim=0)[3]
    target = torch.tensor([0.0, 1.0])
    brier = torch.sum((F.softmax(row[[0, 3]], dim=0) - target).square())
    torch.testing.assert_close(result.restricted_loss, restricted)
    torch.testing.assert_close(result.full_vocab_loss, full)
    torch.testing.assert_close(result.brier_loss, brier)
    torch.testing.assert_close(result.loss, restricted + full + 0.5 * brier)


@pytest.mark.parametrize(
    ("allowed_ids", "target", "expected_target"),
    [
        ((0, 3), (0.0, 1.0), (0.05, 0.95)),
        ((0, 1, 2, 3), (0.0, 0.0, 1.0, 0.0), (0.025, 0.025, 0.925, 0.025)),
    ],
)
def test_one_hot_distribution_smoothing_uses_allowed_support_for_ce_and_keeps_hard_brier(
    allowed_ids, target, expected_target
):
    logits = torch.tensor([[[0.2, 1.5, -0.4, 0.7]]], requires_grad=True)
    mask = torch.tensor([[True]])
    example = _example(_question(0, allowed_ids, DistributionLabel(target)))

    result = decision_label_loss(
        logits,
        [example],
        mask,
        label_softmax="full",
        brier_weight=0.5,
        hard_label_smoothing=0.1,
    )
    result.loss.backward()

    full_logprobs = F.log_softmax(logits.detach()[0, 0], dim=0)
    smoothed = torch.tensor(expected_target)
    hard = torch.tensor(target)
    restricted_probs = F.softmax(logits.detach()[0, 0, list(allowed_ids)], dim=0)
    torch.testing.assert_close(
        result.full_vocab_loss, -(smoothed * full_logprobs[list(allowed_ids)]).sum()
    )
    torch.testing.assert_close(
        result.brier_loss, (restricted_probs - hard).square().sum()
    )
    assert logits.grad is not None
    gold_id = allowed_ids[target.index(1.0)]
    assert logits.grad[0, 0, gold_id] < 0
    for token_id in set(range(logits.shape[-1])) - set(allowed_ids):
        assert logits.grad[0, 0, token_id] > 0


def test_hidden_one_hot_distribution_smoothing_matches_dense_loss():
    torch.manual_seed(123)
    hidden = torch.randn(1, 1, 3, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(3, 5, dtype=torch.float64)
    example = _example(_question(0, (0, 3), DistributionLabel((0.0, 1.0))))
    mask = torch.tensor([[True]])
    kwargs = {
        "label_softmax": "full",
        "brier_weight": 0.1,
        "hard_label_smoothing": 0.1,
    }
    dense = decision_label_loss(
        F.linear(hidden, head.weight, head.bias), [example], mask, **kwargs
    )
    selected = decision_label_loss_from_hidden(
        hidden,
        head,
        [example],
        mask,
        linear_token_loss=_dense_linear_token_loss,
        **kwargs,
    )
    torch.testing.assert_close(selected.loss, dense.loss)
    torch.testing.assert_close(selected.full_vocab_loss, dense.full_vocab_loss)
    torch.testing.assert_close(selected.brier_loss, dense.brier_loss)
    unsmoothed = decision_label_loss(
        F.linear(hidden, head.weight, head.bias),
        [example],
        mask,
        label_softmax="full",
        brier_weight=0.1,
    )
    assert not torch.allclose(dense.loss, unsmoothed.loss)


def test_sparse_one_hot_distribution_smoothing_covers_the_full_allowed_support():
    logits = torch.tensor([[[0.2, 1.5, -0.4, 0.7, 1.1]]], requires_grad=True)
    mask = torch.tensor([[True]])
    example = _example(
        _question(0, (0, 1, 2, 3), DistributionLabel((1.0,), candidate_indices=(3,)))
    )
    smoothed = decision_label_loss(
        logits,
        [example],
        mask,
        label_softmax="full",
        brier_weight=0.0,
        hard_label_smoothing=0.1,
    )
    baseline = decision_label_loss(
        logits, [example], mask, label_softmax="full", brier_weight=0.0
    )
    smoothed.loss.backward()

    expected = torch.tensor([0.025, 0.025, 0.025, 0.925])
    logprobs = F.log_softmax(logits.detach()[0, 0], dim=0)
    torch.testing.assert_close(
        smoothed.full_vocab_loss, -(expected * logprobs[:4]).sum()
    )
    assert not torch.allclose(smoothed.loss, baseline.loss)
    assert logits.grad is not None
    assert logits.grad[0, 0, 3] < 0
    assert logits.grad[0, 0, 4] > 0


def test_hard_label_smoothing_does_not_change_soft_distribution_targets():
    logits = torch.tensor([[[0.2, 1.5, -0.4, 0.7]]])
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 3), DistributionLabel((0.25, 0.75))))
    baseline = decision_label_loss(
        logits, [example], mask, label_softmax="full", brier_weight=0.1
    )
    smoothed = decision_label_loss(
        logits,
        [example],
        mask,
        label_softmax="full",
        brier_weight=0.1,
        hard_label_smoothing=0.1,
    )
    torch.testing.assert_close(smoothed.loss, baseline.loss)


def test_zero_hard_label_smoothing_preserves_one_hot_distribution_loss():
    logits = torch.tensor([[[0.2, 1.5, -0.4, 0.7]]])
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 3), DistributionLabel((0.0, 1.0))))
    baseline = decision_label_loss(logits, [example], mask, label_softmax="full")
    zero = decision_label_loss(
        logits,
        [example],
        mask,
        label_softmax="full",
        hard_label_smoothing=0.0,
    )
    torch.testing.assert_close(zero.loss, baseline.loss)


def test_distribution_uses_restricted_kl_and_full_vocab_soft_ce():
    logits = torch.tensor([[[0.3, -0.2, 1.1, 0.0]]])
    mask = torch.tensor([[True]])
    probabilities = (0.25, 0.75)
    example = _example(_question(0, (1, 2), DistributionLabel(probabilities)))

    result = decision_label_loss(logits, [example], mask, brier_weight=0.0)

    row = logits[0, 0]
    target = torch.tensor(probabilities)
    candidate_logprobs = F.log_softmax(row[[1, 2]], dim=0)
    expected_kl = torch.sum(target * (target.log() - candidate_logprobs))
    expected_full = -(target * F.log_softmax(row, dim=0)[torch.tensor([1, 2])]).sum()
    torch.testing.assert_close(result.restricted_loss, expected_kl)
    torch.testing.assert_close(result.full_vocab_loss, expected_full)
    torch.testing.assert_close(result.loss, expected_kl + expected_full)


def test_sparse_distribution_full_ce_keeps_residual_other_vocab_mass():
    logits = torch.tensor([[[0.3, -0.2, 1.1, 0.0]]])
    mask = torch.tensor([[True]])
    target = DistributionLabel(
        (0.25, 0.5), candidate_indices=(1, 2), other_probability=0.25
    )
    example = _example(_question(0, (0, 1, 2), target))

    result = decision_label_loss(
        logits, [example], mask, label_softmax="full", brier_weight=0.0
    )

    logprobs = F.log_softmax(logits[0, 0], dim=0)
    candidate_mass = logprobs[torch.tensor([1, 2])].exp().sum()
    expected = -(torch.tensor([0.25, 0.5]) * logprobs[torch.tensor([1, 2])]).sum()
    expected = expected - 0.25 * torch.log1p(-candidate_mass)
    torch.testing.assert_close(result.full_vocab_loss, expected)
    torch.testing.assert_close(result.loss, expected)


def test_sparse_distribution_residual_requires_full_softmax():
    logits = torch.zeros((1, 1, 4))
    mask = torch.tensor([[True]])
    target = DistributionLabel((0.75,), candidate_indices=(1,), other_probability=0.25)
    example = _example(_question(0, (0, 1), target))

    with pytest.raises(ValueError, match="requires label_softmax=full"):
        decision_label_loss(logits, [example], mask, label_softmax="both")


def test_sparse_residual_is_finite_when_candidate_mass_rounds_to_one():
    logits = torch.tensor([[[0.0, 100.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    target = DistributionLabel((0.5,), candidate_indices=(1,), other_probability=0.5)
    example = _example(_question(0, (0, 1), target))
    result = decision_label_loss(
        logits,
        [example],
        torch.tensor([[True]]),
        label_softmax="full",
        brier_weight=0.0,
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_hidden_sparse_distribution_residual_matches_dense_path():
    torch.manual_seed(7)
    hidden = torch.randn((1, 1, 3))
    head = torch.nn.Linear(3, 5)
    mask = torch.tensor([[True]])
    target = DistributionLabel((0.5,), candidate_indices=(1,), other_probability=0.5)
    example = _example(_question(0, (0, 1), target))
    logits = F.linear(hidden, head.weight, head.bias)

    expected = decision_label_loss(
        logits, [example], mask, label_softmax="full", brier_weight=0.0
    )
    result = decision_label_loss_from_hidden(
        hidden,
        head,
        [example],
        mask,
        linear_token_loss=_dense_linear_token_loss,
        label_softmax="full",
        brier_weight=0.0,
    )

    torch.testing.assert_close(result.full_vocab_loss, expected.full_vocab_loss)
    torch.testing.assert_close(result.loss, expected.loss)


def test_distribution_one_hot_full_ce_matches_hard_label():
    logits = torch.tensor([[[0.3, -0.2, 1.1, 0.0]]])
    mask = torch.tensor([[True]])
    distribution = _example(_question(0, (1, 2), DistributionLabel((0.0, 1.0))))
    hard = _example(_question(0, (1, 2), HardLabel(1)))

    soft_result = decision_label_loss(logits, [distribution], mask, brier_weight=0.0)
    hard_result = decision_label_loss(logits, [hard], mask, brier_weight=0.0)

    torch.testing.assert_close(soft_result.full_vocab_loss, hard_result.full_vocab_loss)
    torch.testing.assert_close(soft_result.loss, hard_result.loss)


def test_distribution_full_ce_has_soft_target_gradient_and_ignores_zero_mass():
    logits = torch.tensor([[[0.7, -0.3, 1.2, 0.1]]], requires_grad=True)
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 2, 3), DistributionLabel((0.25, 0.75, 0.0))))

    result = decision_label_loss(
        logits, [example], mask, label_softmax="full", brier_weight=0.0
    )
    result.loss.backward()

    target = torch.tensor([0.25, 0.75, 0.0])
    expected = -(
        target * F.log_softmax(logits.detach()[0, 0], 0)[torch.tensor([0, 2, 3])]
    ).sum()
    torch.testing.assert_close(result.full_vocab_loss, expected)
    assert logits.grad is not None
    assert logits.grad[0, 0, 0] != 0
    assert logits.grad[0, 0, 2] != 0


def test_distribution_kl_accepts_zero_probability_alternatives():
    logits = torch.tensor([[[0.1, 0.2, 1.5]]])
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 1, 2), DistributionLabel((0.0, 0.0, 1.0))))

    result = decision_label_loss(logits, [example], mask, brier_weight=0)

    assert torch.isfinite(result.loss)
    torch.testing.assert_close(
        result.restricted_loss, -F.log_softmax(logits[0, 0], 0)[2]
    )


def test_set_uses_allowed_mass_and_detached_best_member_full_vocab_ce():
    logits = torch.tensor([[[1.0, -1.0, 0.4, 2.0]]])
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 1, 2), SetLabel((0, 2))))

    result = decision_label_loss(logits, [example], mask, brier_weight=0.25)

    restricted = F.softmax(logits[0, 0, [0, 1, 2]], dim=0)
    mass = restricted[[0, 2]].sum()
    torch.testing.assert_close(result.restricted_loss, -mass.log())
    torch.testing.assert_close(
        result.full_vocab_loss, -F.log_softmax(logits[0, 0], 0)[0]
    )
    torch.testing.assert_close(result.brier_loss, (1.0 - mass).square())
    torch.testing.assert_close(
        result.loss,
        -mass.log() - F.log_softmax(logits[0, 0], 0)[0] + 0.25 * (1.0 - mass).square(),
    )


def test_set_mass_uses_logsumexp_for_extremely_unlikely_correct_labels():
    logits = torch.tensor([[[1000.0, -1000.0, 999.0]]], requires_grad=True)
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 1, 2), SetLabel((1,))))

    result = decision_label_loss(
        logits, [example], mask, label_softmax="restricted", brier_weight=0
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_set_mass_is_zero_when_every_allowed_label_is_correct():
    logits = torch.tensor([[[0.3, 1.2, -0.1]]])
    example = _example(_question(0, (0, 1, 2), SetLabel((0, 1, 2))))

    result = decision_label_loss(
        logits,
        [example],
        torch.tensor([[True]]),
        label_softmax="restricted",
        brier_weight=0,
    )

    torch.testing.assert_close(result.restricted_loss, torch.zeros(()))
    torch.testing.assert_close(result.brier_loss, torch.zeros(()))


def test_full_softmax_mode_excludes_restricted_primary_for_hard_targets():
    logits = torch.tensor([[[0.1, 1.2, -0.2]]])
    mask = torch.tensor([[True]])
    example = _example(_question(0, (0, 2), HardLabel(0)))

    result = decision_label_loss(
        logits, [example], mask, label_softmax="full", brier_weight=0
    )

    torch.testing.assert_close(result.restricted_loss, torch.zeros(()))
    torch.testing.assert_close(
        result.full_vocab_loss, -F.log_softmax(logits[0, 0], 0)[0]
    )


def test_question_then_example_average_applies_source_weights():
    logits = torch.zeros(2, 2, 4)
    mask = torch.tensor([[True, True], [True, False]])
    examples = [
        _example(
            _question(0, (0, 1), HardLabel(0)),
            _question(1, (2, 3), HardLabel(1)),
            weight=2.0,
        ),
        _example(_question(0, (0, 1, 2, 3), HardLabel(2)), weight=0.5),
    ]

    result = decision_label_loss(
        logits, examples, mask, label_softmax="restricted", brier_weight=0
    )

    expected = (2.0 * math.log(2) + 0.5 * math.log(4)) / 2
    torch.testing.assert_close(result.loss, torch.tensor(expected))


def test_loss_only_backpropagates_through_supervised_label_positions():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    result = decision_label_loss(
        logits,
        [_example(_question(1, (1, 3), HardLabel(0)))],
        mask,
        brier_weight=0,
    )
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


def test_loss_rejects_unsupervised_label_position_and_invalid_target_alignment():
    logits = torch.zeros(1, 2, 3)
    question = _question(1, (0, 1), DistributionLabel((0.5, 0.5)))
    with pytest.raises(RuntimeError, match="supervised"):
        decision_label_loss(logits, [_example(question)], torch.tensor([[True, False]]))
    invalid = _example(_question(0, (0, 1), DistributionLabel((1.0,))))
    with pytest.raises(ValueError, match="align"):
        decision_label_loss(logits, [invalid], torch.tensor([[True, False]]))


@pytest.mark.parametrize(
    "question",
    [
        _question(True, (0, 1), HardLabel(0)),
        _question(0, (True, 1), HardLabel(0)),
        _question(0, (0, 1), HardLabel(True)),
        _question(0, (0, 1), SetLabel((True,))),
    ],
)
def test_direct_target_contract_rejects_boolean_indices(question):
    with pytest.raises(ValueError):
        decision_label_loss(
            torch.zeros(1, 1, 3), [_example(question)], torch.tensor([[True]])
        )


def test_loss_rejects_duplicate_question_positions():
    duplicate = _example(
        _question(0, (0, 1), HardLabel(0)), _question(0, (1, 2), HardLabel(1))
    )
    with pytest.raises(ValueError, match="distinct"):
        decision_label_loss(torch.zeros(1, 1, 3), [duplicate], torch.tensor([[True]]))


def test_per_hard_target_smoothing_matches_global_ce_and_keeps_one_hot_brier():
    logits = torch.tensor([[[0.2, -0.7, 1.3, 0.4, 0.9]]], dtype=torch.float64)
    mask = torch.tensor([[True]])
    global_smoothed = decision_label_loss(
        logits,
        [_example(_question(0, (0, 2, 4), HardLabel(1)))],
        mask,
        label_softmax="full",
        brier_weight=0.1,
        hard_label_smoothing=0.1,
    )
    per_target = decision_label_loss(
        logits,
        [_example(_question(0, (0, 2, 4), HardLabel(1, smoothing=0.1)))],
        mask,
        label_softmax="full",
        brier_weight=0.1,
        hard_label_smoothing=0.0,
    )
    torch.testing.assert_close(per_target.loss, global_smoothed.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        per_target.full_vocab_loss, global_smoothed.full_vocab_loss, rtol=0, atol=0
    )
    torch.testing.assert_close(
        per_target.brier_loss, global_smoothed.brier_loss, rtol=0, atol=0
    )


def test_mixed_hard_target_smoothing_is_finite_with_global_smoothing_disabled():
    logits = torch.tensor([[[0.1, 0.4, -0.3]], [[0.1, 0.4, -0.3]]], dtype=torch.float64)
    result = decision_label_loss(
        logits,
        [
            _example(_question(0, (0, 1, 2), HardLabel(1))),
            _example(_question(0, (0, 1, 2), HardLabel(1, smoothing=0.1))),
        ],
        torch.tensor([[True], [True]]),
        label_softmax="full",
        brier_weight=0.1,
        hard_label_smoothing=0.0,
    )
    assert torch.isfinite(result.loss)
