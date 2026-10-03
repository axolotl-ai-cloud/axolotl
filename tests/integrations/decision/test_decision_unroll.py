"""CPU contracts for the proposed, not-yet-wired decision recurrence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.decision.unroll import run_decision_unroll


def _run(*, steps: int, grad_through_steps: bool, self_conditioning: bool):
    state = torch.tensor([[9, 3, 9, 7]])
    weight = torch.tensor(2.0, requires_grad=True)
    calls = []

    def forward(current, conditioning, mask):
        calls.append((current.clone(), conditioning, mask))
        logits = torch.nn.functional.one_hot(current, 12).float() * weight
        if conditioning is not None:
            logits = logits + conditioning
        return SimpleNamespace(logits=logits)

    outputs, logits, final = run_decision_unroll(
        state=state,
        steps=steps,
        grad_through_steps=grad_through_steps,
        supports_self_conditioning=self_conditioning,
        conditioning_mask=torch.tensor([[False, True, False, True]]),
        forward_step=forward,
        logits_from_outputs=lambda output: output.logits,
    )
    return state, weight, calls, outputs, logits, final


@pytest.mark.parametrize("steps", [2, 3])
def test_decision_recurrence_holds_noisy_labels_and_pinned_slots(steps):
    state, _weight, calls, _outputs, _logits, final = _run(
        steps=steps, grad_through_steps=False, self_conditioning=True
    )

    assert len(calls) == steps
    assert all(torch.equal(call[0], state) for call in calls)
    assert torch.equal(final, state)


@pytest.mark.parametrize("steps", [2, 3])
def test_decision_recurrence_updates_only_free_slots_between_reads(steps):
    state = torch.tensor([[9, 4, 7]])
    update_mask = torch.tensor([[True, False, False]])
    seen = []

    def forward(current, *_args):
        seen.append(current.clone())
        logits = torch.nn.functional.one_hot((current + 1) % 16, 16).float()
        return type("Output", (), {"logits": logits})()

    _, _, final = run_decision_unroll(
        state=state,
        steps=steps,
        grad_through_steps=False,
        supports_self_conditioning=False,
        conditioning_mask=None,
        forward_step=forward,
        logits_from_outputs=lambda output: output.logits,
        update_state=lambda current, logits, mask: torch.where(
            mask, logits.argmax(-1), current
        ),
        update_mask=update_mask,
    )
    assert len(seen) == steps
    assert torch.equal(seen[0][0, 1:], state[0, 1:])
    assert torch.equal(final[0, 1:], state[0, 1:])
    assert final[0, 0] != state[0, 0]
    assert final.tolist() == [[9 + steps - 1, 4, 7]]


def test_decision_recurrence_k1_matches_one_direct_read():
    state, _weight, calls, outputs, logits, final = _run(
        steps=1, grad_through_steps=False, self_conditioning=False
    )

    assert len(calls) == 1
    torch.testing.assert_close(outputs.logits, logits)
    torch.testing.assert_close(final, state)
    assert calls[0][1] is None


def test_non_self_conditioning_recurrence_never_receives_logits():
    _state, _weight, calls, _outputs, _logits, _final = _run(
        steps=3, grad_through_steps=False, self_conditioning=False
    )

    assert [call[1] for call in calls] == [None, None, None]


def test_decision_recurrence_detaches_intermediate_conditioning():
    _state, weight, calls, _outputs, logits, _final = _run(
        steps=2, grad_through_steps=False, self_conditioning=True
    )
    assert calls[1][1] is not None
    assert not calls[1][1].requires_grad
    logits.sum().backward()
    assert weight.grad is not None


def test_decision_recurrence_keeps_differentiable_conditioning_when_enabled():
    _state, weight, calls, _outputs, logits, _final = _run(
        steps=2, grad_through_steps=True, self_conditioning=True
    )
    assert calls[1][1] is not None
    assert calls[1][1].requires_grad
    logits.sum().backward()
    assert weight.grad is not None
    assert weight.grad > 0


def test_decision_recurrence_rejects_grad_through_without_self_conditioning():
    with pytest.raises(ValueError, match="self-conditioning"):
        _run(steps=2, grad_through_steps=True, self_conditioning=False)
