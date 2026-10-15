"""Multimodal processing strategy for CohereCompass VL checkpoints."""

from torch import Tensor

from axolotl.processing_strategies import (
    ProcessingStrategy,
    RoleBoundary,
    _encode_markers,
)

# The chat template writes the assistant turn as "chatbot"; axolotl's role vocabulary calls it assistant.
_ROLE_START_MARKERS = {
    "system": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
    "user": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
    "assistant": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
}

_TURN_END_MARKER = "<|END_OF_TURN_TOKEN|>"


class CohereCompassProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for CohereCompass (Cohere ``<|START_OF_TURN_TOKEN|>`` markers)."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, [_TURN_END_MARKER])
        if not end:
            return []
        end_ids = end[0]

        boundaries = []
        for role, marker in _ROLE_START_MARKERS.items():
            start = _encode_markers(tok, [marker])
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = super().process_labels(input_ids)
        for attr in ("vision_start_token_id", "vision_end_token_id"):
            token_id = getattr(self.processor, attr, None)
            if token_id is not None:
                labels[input_ids == token_id] = -100
        return labels
