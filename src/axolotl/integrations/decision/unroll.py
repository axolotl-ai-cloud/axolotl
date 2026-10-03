"""Decision recurrence which holds labels noised until the final read."""

from __future__ import annotations

import torch


def run_decision_unroll(
    *,
    state: torch.Tensor,
    steps: int,
    grad_through_steps: bool,
    supports_self_conditioning: bool,
    conditioning_mask: torch.Tensor | None,
    forward_step,
    logits_from_outputs,
    update_state=None,
    update_mask=None,
    forward_final=None,
    final_logits_from_outputs=None,
):
    if steps < 1:
        raise ValueError("decision unroll steps must be positive")
    if grad_through_steps and not supports_self_conditioning:
        raise ValueError("decision grad-through-steps requires self-conditioning")
    conditioning = None
    for index in range(steps):
        final = index + 1 == steps
        if final and forward_final is not None:
            outputs = forward_final(state, conditioning, conditioning_mask)
            extract = final_logits_from_outputs or logits_from_outputs
            return outputs, extract(outputs), state
        if final or grad_through_steps:
            outputs = forward_step(state, conditioning, conditioning_mask)
            logits = logits_from_outputs(outputs)
        else:
            with torch.no_grad():
                outputs = forward_step(state, conditioning, conditioning_mask)
                logits = logits_from_outputs(outputs)
        if final:
            return outputs, logits, state
        if update_state is not None and update_mask is not None:
            state = update_state(state, logits, update_mask)
        if supports_self_conditioning:
            conditioning = logits if grad_through_steps else logits.detach()
    raise RuntimeError("decision unroll did not produce final logits")
