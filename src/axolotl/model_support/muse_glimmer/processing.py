"""Multimodal processing strategy for Muse Glimmer."""

from torch import Tensor

from axolotl.processing_strategies import (
    ProcessingStrategy,
    RoleBoundary,
    _encode_markers,
)

IGNORE_INDEX = -100

# Stops before the recipient: add_generation_prompt emits only "<|start|>assistant",
# so the model generates " to=user<|message|>" and it must stay inside the trained span.
_ASSISTANT_START_MARKER = "<|start|>assistant"

# Assistant turns close with either "<|eot|>" or "<|eom|>", so the span ends on the next
# "<|start|>" to cover both. That leaves the terminator inside the span body, where
# `_gate_terminators` re-applies train_on_eos to it.
_TURN_START_MARKER = "<|start|>"
_TURN_END_MARKER = "<|eot|>"
_TERMINATOR_MARKERS = (_TURN_END_MARKER, "<|eom|>")

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

    def _gate_terminators(self, labels: Tensor, input_ids: Tensor) -> Tensor:
        """Apply train_on_eos to the assistant terminators the scanner treats as content.

        Only "none" and "last" need it. Other roles' terminators reach this point only
        when train_on_inputs or roles_to_train trains them, and dropping those is what
        the text-only ChatTemplateStrategy does too.
        """
        if self.train_on_eos not in ("none", "last"):
            return labels

        marker_ids = [
            ids[0]
            for ids in _encode_markers(
                self.processor.tokenizer, list(_TERMINATOR_MARKERS)
            )
            if len(ids) == 1
        ]
        if not marker_ids:
            return labels

        is_terminator = input_ids == marker_ids[0]
        for token_id in marker_ids[1:]:
            is_terminator = is_terminator | (input_ids == token_id)
        drop = is_terminator & (labels != IGNORE_INDEX)

        if self.train_on_eos == "last":
            for row in drop:
                kept = row.nonzero()
                if kept.numel():
                    row[kept[-1]] = False

        labels[drop] = IGNORE_INDEX
        return labels

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = super().process_labels(input_ids)
        for attr in _MEDIA_TOKEN_ID_ATTRS:
            token_id = getattr(self.processor, attr, None)
            if token_id is not None:
                labels[input_ids == token_id] = IGNORE_INDEX
        return self._gate_terminators(labels, input_ids)
