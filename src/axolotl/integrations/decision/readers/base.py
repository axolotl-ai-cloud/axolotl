"""Shared typed results and validation for decision readers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch

from ..records import DecisionCanvas


@dataclass(frozen=True)
class ReadDiagnostics:
    """Inputs and recurrence facts retained for an optional parity fixture."""

    noise_seed: int | None
    noise_kind: str
    update_policy: str
    steps: int
    forward_count: int
    initial_canvas_ids: torch.Tensor
    final_canvas_ids: torch.Tensor
    slot_init_policy: str = "prepared_canvas_v0"


@dataclass(frozen=True)
class DecisionRead:
    """Full-vocabulary and candidate-restricted scores for one decision canvas."""

    question_ids: tuple[str, ...]
    label_positions: torch.Tensor
    allowed_ids: torch.Tensor
    candidate_mask: torch.Tensor
    full_vocab_logprobs: torch.Tensor
    restricted_probs: torch.Tensor
    diagnostics: ReadDiagnostics | None = None


def validate_canvas(canvas: DecisionCanvas) -> int:
    """Validate the model-agnostic canvas invariants before any model execution."""

    width = len(canvas.canvas_ids)
    if not canvas.prompt_ids or width < 2:
        raise ValueError("a decision canvas requires a prompt and token space")
    if (
        isinstance(canvas.template_length, bool)
        or not isinstance(canvas.template_length, Integral)
        or not 0 < canvas.template_length < width
    ):
        raise ValueError("template_length must end before the turn-close token")
    for tokens in (canvas.prompt_ids, canvas.canvas_ids, *canvas.allowed_ids):
        if any(
            isinstance(token, bool) or not isinstance(token, Integral) or token < 0
            for token in tokens
        ):
            raise ValueError("decision token IDs must be nonnegative integers")
    if len(set(canvas.question_ids)) != len(canvas.question_ids) or any(
        not isinstance(key, str) or not key for key in canvas.question_ids
    ):
        raise ValueError("question IDs must be distinct nonempty strings")
    fields: tuple[tuple[str, Sequence[object]], ...] = (
        ("pinned_mask", canvas.pinned_mask),
        ("semantic_mask", canvas.semantic_mask),
        ("slot_mask", canvas.slot_mask),
    )
    for name, values in fields:
        if len(values) != width:
            raise ValueError(f"{name} must have one value for each canvas token")
    if canvas.prompt_slot_mask:
        if len(canvas.prompt_slot_mask) != len(canvas.prompt_ids):
            raise ValueError(
                "prompt_slot_mask must have one value for each prompt token"
            )
        if any(not isinstance(value, bool) for value in canvas.prompt_slot_mask):
            raise TypeError("prompt_slot_mask values must be bool")
    question_count = len(canvas.label_positions)
    if question_count == 0:
        raise ValueError("a decision canvas must contain at least one label position")
    if not (
        len(canvas.allowed_ids) == len(canvas.question_ids) == question_count
        and len(canvas.targets) in (0, question_count)
    ):
        raise ValueError("question metadata must align with label_positions")
    if canvas.ordinal_metadata and len(canvas.ordinal_metadata) != question_count:
        raise ValueError("ordinal metadata must align with canvas questions")
    seen_positions: set[int] = set()
    for index, (position, candidates) in enumerate(
        zip(canvas.label_positions, canvas.allowed_ids, strict=True)
    ):
        if (
            isinstance(position, bool)
            or not isinstance(position, Integral)
            or position < 0
            or position >= canvas.template_length
        ):
            raise ValueError(
                f"label position {position} is outside the decision canvas"
            )
        if position in seen_positions:
            raise ValueError("each question must have a distinct label position")
        if canvas.slot_mask[position]:
            raise ValueError("latent slots cannot be label positions")
        if not canvas.semantic_mask[position]:
            raise ValueError("label positions must be semantically valid")
        if not candidates:
            raise ValueError(f"question {index} has no allowed label token")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"question {index} repeats an allowed label token")
        if canvas.ordinal_metadata:
            _validate_ordinal_metadata(canvas.ordinal_metadata[index], len(candidates))
        seen_positions.add(position)
    return width


def restricted_probabilities(
    full_vocab_logprobs: torch.Tensor, allowed_ids: Sequence[Sequence[int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a per-question candidate softmax without losing full-vocabulary scores."""

    if full_vocab_logprobs.ndim != 2:
        raise ValueError("full_vocab_logprobs must be [questions, vocabulary]")
    question_count, vocab_size = full_vocab_logprobs.shape
    if question_count != len(allowed_ids):
        raise ValueError("allowed_ids must contain one candidate sequence per question")
    maximum = max(len(ids) for ids in allowed_ids)
    ids = torch.zeros(
        (question_count, maximum), dtype=torch.long, device=full_vocab_logprobs.device
    )
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for row, candidates in enumerate(allowed_ids):
        if any(token < 0 or token >= vocab_size for token in candidates):
            raise ValueError("allowed label token ID is outside the model vocabulary")
        candidate_ids = torch.as_tensor(
            candidates, dtype=torch.long, device=full_vocab_logprobs.device
        )
        ids[row, : candidate_ids.numel()] = candidate_ids
        valid[row, : candidate_ids.numel()] = True
    selected = full_vocab_logprobs.gather(1, ids)
    selected = selected.masked_fill(~valid, -torch.inf)
    probabilities = torch.softmax(selected, dim=-1).masked_fill(~valid, 0.0)
    return ids, valid, probabilities


def _validate_ordinal_metadata(value: object, count: int) -> None:
    if value is None:
        return
    from ..records import OrdinalMetadata

    if not isinstance(value, OrdinalMetadata):
        raise TypeError("ordinal metadata must be OrdinalMetadata or None")
    fields = (value.levels, value.source_ids, value.candidate_ranks)
    if count < 2:
        raise ValueError("ordinal metadata requires at least two candidates")
    if any(len(field) != count for field in fields):
        raise ValueError("ordinal metadata must align with allowed candidates")
    if any(not isinstance(level, str) or not level for level in value.levels) or any(
        not isinstance(source_id, str) or not source_id
        for source_id in value.source_ids
    ):
        raise ValueError("ordinal levels and source IDs must be nonempty strings")
    ranks = tuple(value.candidate_ranks)
    if any(isinstance(rank, bool) or not isinstance(rank, Integral) for rank in ranks):
        raise TypeError("ordinal candidate ranks must be integers")
    if set(ranks) != set(range(count)):
        raise ValueError("ordinal candidate ranks must be a zero-based bijection")
