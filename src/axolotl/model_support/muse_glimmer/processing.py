"""Multimodal processing strategy for Muse Glimmer."""

from torch import Tensor

from axolotl.processing_strategies import (
    ProcessingStrategy,
    RoleBoundary,
    _encode_markers,
)

# Stops before the recipient: add_generation_prompt emits only "<|start|>assistant",
# so the model generates " to=user<|message|>" and it must stay inside the trained span.
_ASSISTANT_START_MARKER = "<|start|>assistant"

# Assistant turns close with "<|eom|>" rather than "<|eot|>" on a non-"user" recipient,
# `end_turn: false`, or a non-final tool call, so the span ends on the next "<|start|>"
# to cover both. Cost: train_on_eos cannot gate the assistant terminator.
_TURN_START_MARKER = "<|start|>"
_TURN_END_MARKER = "<|eot|>"

_ROLE_START_MARKERS = {
    "system": "<|start|>system<|message|>",
    "user": "<|start|>user<|message|>",
    "tool": "<|start|>tool",
}

_MEDIA_TOKEN_ID_ATTRS = (
    "image_token_id",
    "image_start_token_id",
    "image_end_token_id",
    "video_token_id",
)


class MuseGlimmerProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Muse Glimmer."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, [_TURN_END_MARKER])
        turn_start = _encode_markers(tok, [_TURN_START_MARKER])
        assistant_start = _encode_markers(tok, [_ASSISTANT_START_MARKER])
        if not end or not turn_start or not assistant_start:
            return []
        end_ids = end[0]

        boundaries = []
        for role, marker in _ROLE_START_MARKERS.items():
            start = _encode_markers(tok, [marker])
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        boundaries.append(
            RoleBoundary(
                role="assistant",
                start_tokens=assistant_start[0],
                end_tokens=turn_start[0],
                include_end=False,
            )
        )
        return boundaries

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = super().process_labels(input_ids)
        for attr in _MEDIA_TOKEN_ID_ATTRS:
            token_id = getattr(self.processor, attr, None)
            if token_id is not None:
                labels[input_ids == token_id] = -100
        return labels
