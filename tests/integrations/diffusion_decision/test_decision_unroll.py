"""CPU contract for the non-self-conditioning decision recurrence."""

from __future__ import annotations

import pytest
import torch

from axolotl.core.trainers.diffusion_lm.unroll import run_unroll


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_decision_recurrence_updates_only_free_slots_between_reads(steps):
    state = torch.tensor([[9, 4, 7]])
    update_mask = torch.tensor([[True, False, False]])
    seen = []

    def forward(current, *_args):
        seen.append(current.clone())
        logits = torch.nn.functional.one_hot((current + 1) % 16, 16).float()
        return type("Output", (), {"logits": logits})()

    _, _, final = run_unroll(
        state=state,
        update_mask=update_mask,
        steps=steps,
        grad_through_steps=False,
        supports_self_conditioning=False,
        k1_conditioning_mask=None,
        recurrent_conditioning_mask=None,
        forward_step=forward,
        logits_from_outputs=lambda output: output.logits,
        update_state=lambda current, logits, mask: torch.where(
            mask, logits.argmax(-1), current
        ),
        pilot_for_single_step=False,
    )
    assert len(seen) == steps
    assert torch.equal(final[0, 1:], state[0, 1:])
    assert final.tolist() == [[9 + steps - 1, 4, 7]]
