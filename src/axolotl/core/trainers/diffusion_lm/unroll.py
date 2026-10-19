"""Shared native diffusion recurrence."""

from __future__ import annotations

import torch


def run_unroll(
    *,
    state: torch.Tensor,
    update_mask: torch.Tensor,
    steps: int,
    grad_through_steps: bool,
    supports_self_conditioning: bool,
    k1_conditioning_mask: torch.Tensor | None,
    recurrent_conditioning_mask: torch.Tensor | None,
    forward_step,
    logits_from_outputs,
    update_state,
    forward_final=None,
    final_logits_from_outputs=None,
    pilot_for_single_step: bool = True,
):
    """Run native denoising and return its final outputs, logits, and input state."""

    if steps < 1:
        raise ValueError("unroll steps must be positive")
    if grad_through_steps and not supports_self_conditioning:
        raise ValueError(
            "unroll.grad_through_steps requires a self-conditioning diffusion model"
        )
    if steps == 1 and pilot_for_single_step:
        conditioning = None
        if supports_self_conditioning:
            with torch.no_grad():
                pilot_outputs = forward_step(state, None, None)
            if k1_conditioning_mask is not None:
                conditioning = logits_from_outputs(pilot_outputs).detach()
        final_forward = forward_final or forward_step
        final_logits = final_logits_from_outputs or logits_from_outputs
        outputs = final_forward(state, conditioning, k1_conditioning_mask)
        return outputs, final_logits(outputs), state

    conditioning = None
    for index in range(steps):
        is_final_step = index + 1 == steps
        if is_final_step:
            final_forward = forward_final or forward_step
            final_logits = final_logits_from_outputs or logits_from_outputs
            outputs = final_forward(state, conditioning, recurrent_conditioning_mask)
            logits = final_logits(outputs)
        elif grad_through_steps:
            outputs = forward_step(state, conditioning, recurrent_conditioning_mask)
            logits = logits_from_outputs(outputs)
        else:
            with torch.no_grad():
                outputs = forward_step(state, conditioning, recurrent_conditioning_mask)
                logits = logits_from_outputs(outputs)
        if is_final_step:
            return outputs, logits, state
        conditioning = logits if grad_through_steps else logits.detach()
        state = update_state(state, logits, update_mask)
    raise RuntimeError("unroll did not produce a final step")
