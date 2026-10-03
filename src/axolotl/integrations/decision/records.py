from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class OrdinalMetadata:
    """Source-defined score ordering for one candidate-restricted question."""

    levels: Sequence[str]
    source_ids: Sequence[str]
    candidate_ranks: Sequence[int]


@dataclass(frozen=True)
class DecisionCanvas:
    prompt_ids: Sequence[int]
    canvas_ids: Sequence[int]
    label_positions: Sequence[int]
    allowed_ids: Sequence[Sequence[int]]
    question_ids: Sequence[str]
    targets: Sequence[Any]
    pinned_mask: Sequence[bool]
    semantic_mask: Sequence[bool]
    slot_mask: Sequence[bool]
    template_length: int
    prompt_slot_mask: Sequence[bool] = ()
    ordinal_metadata: Sequence[OrdinalMetadata | None] = ()
