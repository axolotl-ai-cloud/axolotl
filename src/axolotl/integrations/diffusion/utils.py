"""Compatibility imports for diffusion utilities."""

from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    create_bidirectional_attention_mask,
    shift_logits_to_input_positions,
)
from axolotl.core.trainers.diffusion_lm.tokens import resolve_mask_token_id

__all__ = [
    "create_bidirectional_attention_mask",
    "resolve_mask_token_id",
    "shift_logits_to_input_positions",
]
