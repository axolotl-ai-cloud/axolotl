"""Full-sequence decision training runs only unroll passes that can matter."""

from __future__ import annotations

import pytest
import torch

from axolotl.core.trainers.diffusion_lm.unroll import run_unroll
from axolotl.integrations.diffusion_decision import trainer as trainer_module

from tests.integrations.diffusion_decision.test_eval_flattening import _tiny_nemotron
from tests.integrations.diffusion_decision.test_trainer import (
    _FULL,
    _collate,
    _free_slot_canvas,
    _full_sequence_inputs,
    _latent,
    _multistep,
    _SelectedTinyNativeModel,
    _TinyNativeModel,
)

_DECISION = {"labels": {"label_softmax": "both", "brier_weight": 0.1}}


def _recording(model):
    """Record the input ids and grad mode of every forward of ``model``."""

    calls: list[tuple[torch.Tensor, bool]] = []
    original = model.forward

    def forward(*args, **kwargs):
        calls.append((kwargs["input_ids"].detach().clone(), torch.is_grad_enabled()))
        return original(*args, **kwargs)

    model.forward = forward
    return calls


def _every_sampled_pass(monkeypatch):
    """Restore the pre-fix behaviour: run every sampled pass, even dead ones."""

    def run_all(**kwargs):
        if kwargs["steps"] == 1 and not kwargs["update_mask"].any():
            kwargs["steps"] = 2
        return run_unroll(**kwargs)

    monkeypatch.setattr(trainer_module, "run_unroll", run_all)


def _train_step(model, trainer, inputs):
    model.train()
    model.zero_grad(set_to_none=True)
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), grads


@pytest.mark.parametrize("selected", [False, True])
def test_dead_training_pass_is_skipped_with_identical_loss_and_gradients(
    monkeypatch, selected
):
    inputs = _full_sequence_inputs(input_ids=torch.tensor([[1, 40, 5, 2, 60, 7]]))

    with monkeypatch.context() as patch:
        _every_sampled_pass(patch)
        reference_model = _tiny_nemotron()
        if not selected:
            reference_model.supports_selected_logits = False
        reference_calls = _recording(reference_model)
        reference_loss, reference_grads = _train_step(
            reference_model, _multistep(2, k_max=2, decision=_DECISION), inputs
        )

    model = _tiny_nemotron()
    if not selected:
        model.supports_selected_logits = False
    calls = _recording(model)
    logged = []
    monkeypatch.setattr(
        trainer_module.LOG, "info", lambda message, *args: logged.append(message)
    )
    trainer = _multistep(2, k_max=2, decision=_DECISION)
    loss, grads = _train_step(model, trainer, inputs)
    _train_step(model, trainer, inputs)

    # The pre-fix reference ran a no-grad pass on the exact final-pass state.
    assert [grad for _, grad in reference_calls] == [False, True]
    assert torch.equal(reference_calls[0][0], reference_calls[1][0])
    assert [grad for _, grad in calls] == [True, True]
    assert torch.equal(calls[0][0], reference_calls[1][0])
    assert torch.equal(loss, reference_loss)
    assert grads.keys() == reference_grads.keys() and grads
    for name, grad in grads.items():
        assert torch.equal(grad, reference_grads[name]), name
    assert len(logged) == 1 and "only the final" in logged[0]


def test_skipped_pass_matches_a_k_max_one_configuration():
    inputs = _full_sequence_inputs(input_ids=torch.tensor([[1, 40, 5, 2, 60, 7]]))
    single = _tiny_nemotron()
    unrolled = _tiny_nemotron()

    single_loss, single_grads = _train_step(
        single, _multistep(1, k_max=1, decision=_DECISION), inputs
    )
    unrolled_loss, unrolled_grads = _train_step(
        unrolled, _multistep(2, k_max=2, decision=_DECISION), inputs
    )

    assert torch.equal(single_loss, unrolled_loss)
    for name, grad in single_grads.items():
        assert torch.equal(grad, unrolled_grads[name]), name


def test_free_slot_passes_still_run_and_feed_the_final_pass():
    trainer = _multistep(3, decision=_latent("free", free_update_policy="argmax"))
    inputs = _collate(_free_slot_canvas((1,), (9, 3, 4, 0), "q"), layout=_FULL)
    model = _TinyNativeModel()

    trainer._full_sequence_logits(
        model, inputs, trainer._native_spec, trainer._decision_config()
    )

    assert model.forward_grad_enabled == [False, False, True]
    assert model.forward_states[1][0, 1].item() == 10


def _free_slot_inputs():
    return _collate(
        _free_slot_canvas((1, 2), (9, 3, 4, 0), "first"),
        _free_slot_canvas((3,), (8, 5, 6, 0), "second"),
        layout=_FULL,
    )


@pytest.mark.parametrize("model_kind", ["bias", "nemotron"])
@pytest.mark.parametrize("steps", [2, 3])
def test_free_slot_passes_use_selected_logits_with_identical_commits(model_kind, steps):
    def make(selected: bool):
        if model_kind == "nemotron":
            model = _tiny_nemotron()
            if not selected:
                model.supports_selected_logits = False
            return model
        model = (_SelectedTinyNativeModel if selected else _TinyNativeModel)(
            vocab_size=128
        )
        torch.manual_seed(9)
        with torch.no_grad():
            model.logit_bias.copy_(torch.randn(128))
        return model

    results = {}
    for selected in (False, True):
        model = make(selected)
        if model_kind == "nemotron":
            calls = _recording(model)
        trainer = _multistep(
            steps,
            decision={**_DECISION, **_latent("free", free_update_policy="argmax")},
        )
        loss, grads = _train_step(model, trainer, _free_slot_inputs())
        states = (
            [state for state, _ in calls]
            if model_kind == "nemotron"
            else model.forward_states
        )
        results[selected] = (loss, grads, states, model)

    dense_loss, dense_grads, dense_states, _ = results[False]
    loss, grads, states, selected_model = results[True]
    assert len(states) == len(dense_states) == steps
    for state, dense_state in zip(states, dense_states, strict=True):
        assert torch.equal(state, dense_state)
    # Free slots really are committed between passes.
    slots = _free_slot_inputs()["decision_slot_mask"]
    assert not torch.equal(dense_states[0][slots], dense_states[1][slots]) or (
        model_kind == "bias"
    )
    torch.testing.assert_close(loss, dense_loss, rtol=1e-6, atol=1e-6)
    for name, grad in dense_grads.items():
        torch.testing.assert_close(grads[name], grad, rtol=1e-6, atol=1e-6)
    if model_kind == "bias":
        # Every non-final pass projects only the two free slots.
        assert selected_model.selected_shapes[: steps - 1] == [
            torch.Size([1, 2, 128])
        ] * (steps - 1)
