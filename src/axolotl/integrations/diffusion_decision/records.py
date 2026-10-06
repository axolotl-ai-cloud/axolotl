from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence


@dataclass(frozen=True)
class OrdinalMetadata:
    """Source-defined score ordering for one candidate-restricted question."""

    levels: Sequence[str]
    source_ids: Sequence[str]
    candidate_ranks: Sequence[int]

    def __post_init__(self) -> None:
        count = len(self.levels)
        if len(self.source_ids) != count or len(self.candidate_ranks) != count:
            raise ValueError("ordinal metadata must align with allowed candidates")
        if (
            any(not isinstance(item, str) or not item for item in self.levels)
            or any(not isinstance(item, str) or not item for item in self.source_ids)
            or len(set(self.source_ids)) != count
        ):
            raise ValueError(
                "ordinal levels and source IDs must be nonempty and source IDs unique"
            )
        if any(
            isinstance(rank, bool) or not isinstance(rank, Integral)
            for rank in self.candidate_ranks
        ):
            raise ValueError("ordinal candidate ranks must be integers")
        if set(self.candidate_ranks) != set(range(count)):
            raise ValueError("ordinal candidate ranks must be a zero-based bijection")

    def check_candidates(self, count: int) -> None:
        if count < 2:
            raise ValueError("ordinal metadata requires at least two candidates")
        if len(self.levels) != count:
            raise ValueError("ordinal metadata must align with allowed candidates")


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
    image_refs: Sequence[str] = ()
    image_sizes: Sequence[tuple[int, int]] = ()
