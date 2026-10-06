"""Pinned-source objective contracts for diffusion weighting primitives."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer
from axolotl.core.trainers.diffusion_lm.weighting import (
    cart_weights,
    focal_weighted_nll,
    reduce_objective,
    rhine_loo_nll,
    time_weights,
)
from axolotl.model_support.diffusion import ObjectiveReduction, TimeWeighting

_FIXTURE = Path(__file__).with_name("fixtures") / "rhine_loo_ce_26e764.json"


def test_dream_time_and_focal_weights_match_pinned_formula():
    times = torch.tensor([0.2, 0.75])
    assert torch.equal(time_weights(times, TimeWeighting.INV_T), 1 / times)
    assert torch.equal(time_weights(times, TimeWeighting.LINEAR), 1 - times)
    nll = torch.tensor([0.0, 0.5, 2.0])
    expected = 0.3 * (1 - torch.exp(-nll)).pow(1.7) * nll
    assert torch.allclose(focal_weighted_nll(nll, 0.3, 1.7), expected)


def test_cart_isolated_per_document_and_semantic_tail_is_context():
    unmasked = torch.tensor(
        [
            [True, False, True, True],
            [False, True, False, True],
        ]
    )
    weights = cart_weights(unmasked, 0.1)
    positions = torch.arange(4)
    matrix = (
        0.5
        * 0.1
        * 0.9 ** (positions[:, None] - positions[None, :]).abs().sub(1).clamp_min(0)
    )
    matrix.fill_diagonal_(0)
    expected = torch.stack([(matrix * row[None].float()).sum(-1) for row in unmasked])
    assert torch.allclose(weights, expected)
    # The trailing True is a semantic EOS tail, so it contributes to the same
    # logical document. A second packed document cannot alter the first row.
    assert weights[0, 1] > 0
    assert torch.equal(weights[0], cart_weights(unmasked[:1], 0.1)[0])


def test_rhine_loo_matches_pinned_loss_and_gradients():
    fixture = json.loads(_FIXTURE.read_text())
    logits = torch.tensor(fixture["logits"], requires_grad=True)
    targets = torch.tensor(fixture["targets"])
    noisy = torch.tensor(fixture["noisy_ids"])
    times = torch.tensor(fixture["times"])
    support = torch.tensor(fixture["loss_mask"], dtype=torch.bool)
    ours = reduce_objective(
        rhine_loo_nll(logits, targets, noisy, times),
        support,
        ObjectiveReduction.EXAMPLE_MEAN,
        logical_ids=torch.tensor([0, 0, 1, 1]),
        logical_count=2,
    )
    ours.backward()
    ours_grad = logits.grad.detach().clone()

    assert torch.allclose(ours, torch.tensor(fixture["expected_loss"]))
    assert torch.allclose(ours_grad, torch.tensor(fixture["expected_logit_gradients"]))


def test_reductions_preserve_unequal_logical_examples_and_empty_support():
    loss = torch.tensor([1.0, 3.0, 9.0], requires_grad=True)
    support = torch.tensor([True, True, True])
    ids = torch.tensor([0, 0, 1])
    token_mean = reduce_objective(
        loss, support, ObjectiveReduction.SUPERVISED_TOKEN_MEAN
    )
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


def test_uniform_canvas_and_encoder_ar_match_pinned_loss_and_gradients():
    fixture = json.loads(
        (
            Path(__file__).with_name("fixtures") / "reference_uniform_ce_f21252.json"
        ).read_text()
    )
    canvas = torch.tensor(fixture["canvas_logits"], requires_grad=True)
    encoder = torch.tensor(fixture["encoder_logits"], requires_grad=True)
    canvas_targets = torch.tensor(fixture["canvas_targets"])
    canvas_mask = torch.tensor(fixture["canvas_loss_mask"])
    encoder_ids = torch.tensor(fixture["encoder_input_ids"])
    encoder_validity = torch.tensor(fixture["encoder_validity"])
    canvas_terms, ar_terms = [], []
    for row in range(canvas.shape[0]):
        canvas_nll = torch.nn.functional.cross_entropy(
            canvas[row], canvas_targets[row], reduction="none"
        )
        canvas_terms.append(
            reduce_objective(
                canvas_nll,
                canvas_mask[row],
                ObjectiveReduction.SUPERVISED_TOKEN_MEAN,
                denominator=fixture["canvas_denominator"],
            )
        )
        ar_nll = torch.nn.functional.cross_entropy(
            encoder[row, :-1], encoder_ids[row, 1:], reduction="none"
        )
        ar_terms.append(
            reduce_objective(
                ar_nll,
                encoder_validity[row, :-1] & encoder_validity[row, 1:],
                ObjectiveReduction.SUPERVISED_TOKEN_MEAN,
                denominator=fixture["encoder_ar_denominator"],
            )
        )
    diffusion, ar = sum(canvas_terms), sum(ar_terms)
    total = diffusion + ar
    total.backward()
    torch.testing.assert_close(diffusion, torch.tensor(fixture["diffusion_loss"]))
    torch.testing.assert_close(ar, torch.tensor(fixture["encoder_ar_loss"]))
    torch.testing.assert_close(total, torch.tensor(fixture["total_loss"]))
    torch.testing.assert_close(canvas.grad, torch.tensor(fixture["canvas_gradients"]))
    torch.testing.assert_close(encoder.grad, torch.tensor(fixture["encoder_gradients"]))


def test_dream_cart_matches_source_autocast_weights_loss_and_gradients():
    fixture = json.loads(
        (Path(__file__).with_name("fixtures") / "dream_cart_31f94a.json").read_text()
    )
    unmasked = torch.tensor(fixture["unmasked"])
    targets = torch.tensor(fixture["targets"])
    for case in fixture["cases"]:
        logits = torch.tensor(fixture["logits"], requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=case["bf16_autocast"]):
            weights = cart_weights(unmasked, fixture["cart_p"]).masked_fill(unmasked, 0)
            nll = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape_as(targets)
            loss = reduce_objective(
                nll * weights, ~unmasked, ObjectiveReduction.MASKED_TOKEN_MEAN
            )
        loss.backward()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=case["bf16_autocast"]):
            packed_weights = (
                AxolotlDiffusionTrainer._packed_cart_weights(
                    (~unmasked).reshape(1, -1),
                    torch.ones(1, unmasked.numel(), dtype=torch.bool),
                    torch.arange(2)[:, None].expand_as(unmasked).reshape(1, -1),
                    fixture["cart_p"],
                )
                .reshape_as(unmasked)
                .masked_fill(unmasked, 0)
            )
        torch.testing.assert_close(
            packed_weights, torch.tensor(case["weights"]), rtol=0, atol=0
        )
        expected_dtype = torch.bfloat16 if case["bf16_autocast"] else torch.float32
        assert weights.dtype == expected_dtype
        torch.testing.assert_close(
            weights.float(), torch.tensor(case["weights"]), rtol=0, atol=0
        )
        torch.testing.assert_close(loss, torch.tensor(case["loss"]))
        torch.testing.assert_close(logits.grad, torch.tensor(case["logit_gradients"]))
