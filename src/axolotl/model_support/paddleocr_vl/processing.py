"""Multimodal processing strategy for PaddleOCR-VL."""

from axolotl.processing_strategies import (
    ProcessingStrategy,
    RoleBoundary,
    _encode_markers,
)


class PaddleOCRVLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for PaddleOCR-VL."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        assistant_start = _encode_markers(tok, ["Assistant:\n"])
        eos = getattr(tok, "eos_token_id", None)
        if not assistant_start or eos is None:
            return []

        boundaries = []
        # The template writes "User: " but the trailing space BPE-merges into
        # the first content word on text-only turns; match the stable prefix.
        user_start = _encode_markers(tok, ["User:"])
        if user_start:
            boundaries.append(
                RoleBoundary(
                    role="user",
                    start_tokens=user_start[0],
                    end_tokens=assistant_start[0],
                    include_end=False,
                )
            )
        boundaries.append(
            RoleBoundary(
                role="assistant",
                start_tokens=assistant_start[0],
                end_tokens=[eos],
            )
        )
        return boundaries
